from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from mne_lsl.stream import StreamLSL
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from psychopy import core, event, visual

from bci_runtime import (
    apply_runtime_config_overrides,
    resolve_runtime_jaw_classifier,
    resolve_shared_epoch_mi_model,
)
from config import EEGConfig, HingeTaskConfig, LSLConfig, MentalCommandModelConfig, StimConfig
from derick_ml_jawclench import run_visual_jaw_calibration, select_jaw_channel_indices, update_live_jaw_clench_state
from mental_command_worker import canonicalize_channel_name, filter_session, resolve_channel_order
from mi_keyboard_task import run_task as run_keyboard_task


@dataclass(frozen=True)
class PromptBlock:
    prompt: str
    response: str


@dataclass(frozen=True)
class HingeProfile:
    folder_name: str
    name: str
    occupation: str
    age: int
    prompt_1: PromptBlock
    prompt_2: PromptBlock
    picture_1: str
    picture_2: str
    picture_3: str


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"hinge.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"{fname}_hinge.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _sanitize(raw: str) -> str:
    cleaned = "_".join(raw.strip().lower().split())
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    return cleaned.strip("_")


def _prompt_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        p = _sanitize(raw)
        if p:
            return f"{datetime.now().strftime('%m_%d_%y')}_{p}_hinge"
        print("Name cannot be empty.")


def _load_profiles(profiles_dir: str | Path) -> list[HingeProfile]:
    root = Path(profiles_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Profiles directory not found at {root}. Create bci/profiles/<name>/ with pictures and profile_metadata.json."
        )

    profiles: list[HingeProfile] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        metadata_path = folder / "profile_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing profile_metadata.json in {folder}")
        with open(metadata_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        for key in ("picture_1.jpg", "picture_2.jpg", "picture_3.jpg"):
            if not (folder / key).exists():
                raise FileNotFoundError(f"Missing {key} in {folder}")

        prompt_1 = meta.get("prompt_1", {})
        prompt_2 = meta.get("prompt_2", {})
        profiles.append(
            HingeProfile(
                folder_name=folder.name,
                name=str(meta["name"]),
                occupation=str(meta["occupation"]),
                age=int(meta["age"]),
                prompt_1=PromptBlock(prompt=str(prompt_1["prompt"]), response=str(prompt_1["response"])),
                prompt_2=PromptBlock(prompt=str(prompt_2["prompt"]), response=str(prompt_2["response"])),
                picture_1=str((folder / "picture_1.jpg").resolve()),
                picture_2=str((folder / "picture_2.jpg").resolve()),
                picture_3=str((folder / "picture_3.jpg").resolve()),
            )
        )
    if not profiles:
        raise RuntimeError(f"No profile folders found under {root}")
    return profiles


def _fit_image_size(image_path: str, max_w: float, max_h: float) -> tuple[float, float]:
    with Image.open(image_path) as img:
        width_px, height_px = img.size
    if width_px <= 0 or height_px <= 0:
        return max_w, max_h
    scale = min(float(max_w) / float(width_px), float(max_h) / float(height_px))
    return float(width_px) * scale, float(height_px) * scale


@dataclass
class ProfileStimBundle:
    stims: list[object]
    temp_paths: list[Path]


def _load_ui_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ) if bold else (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, *, max_lines: int | None = None) -> str:
    words = str(text).split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        kept = lines[:max_lines]
        kept[-1] = kept[-1].rstrip(" .,;:") + "..."
        lines = kept
    return "\n".join(lines)


def _draw_shadow(base: Image.Image, rect: tuple[int, int, int, int], radius: int, *, blur: int = 22, offset: tuple[int, int] = (0, 14), fill: tuple[int, int, int, int] = (88, 59, 41, 42)) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = rect
    ox, oy = offset
    draw.rounded_rectangle((x0 + ox, y0 + oy, x1 + ox, y1 + oy), radius=radius, fill=fill)
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius=blur)))


def _paste_cover_image(base: Image.Image, image_path: str, rect: tuple[int, int, int, int], radius: int) -> None:
    x0, y0, x1, y1 = rect
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    with Image.open(image_path) as img:
        photo = img.convert("RGB")
    scale = max(float(target_w) / float(photo.width), float(target_h) / float(photo.height))
    resized = photo.resize(
        (max(1, int(round(photo.width * scale))), max(1, int(round(photo.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    mask = Image.new("L", (target_w, target_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, target_w, target_h), radius=radius, fill=255)
    base.paste(cropped, (x0, y0), mask)


def _render_profile_image(profile: HingeProfile) -> Image.Image:
    width_px = 1560
    height_px = 1040
    canvas = Image.new("RGBA", (width_px, height_px), (247, 238, 233, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _load_ui_font(92, bold=True)
    age_font = _load_ui_font(68, bold=True)
    badge_font = _load_ui_font(36, bold=True)
    prompt_font = _load_ui_font(44, bold=True)
    body_font = _load_ui_font(46, bold=False)
    meta_font = _load_ui_font(50, bold=False)
    small_font = _load_ui_font(34, bold=False)

    palette = {
        "page": (247, 238, 233, 255),
        "card": (255, 252, 250, 255),
        "tile": (252, 247, 244, 255),
        "border": (231, 220, 214, 255),
        "accent": (233, 91, 114, 255),
        "accent_soft": (252, 233, 237, 255),
        "text": (58, 42, 34, 255),
        "muted": (112, 95, 84, 255),
        "pill": (69, 58, 52, 180),
    }

    card_rect = (24, 22, width_px - 24, height_px - 22)
    _draw_shadow(canvas, card_rect, 42, blur=24, offset=(0, 14), fill=(91, 71, 62, 34))
    draw.rounded_rectangle(card_rect, radius=42, fill=palette["card"], outline=palette["border"], width=3)

    x0, y0, x1, y1 = card_rect
    inner_x = x0 + 30
    inner_y = y0 + 30
    col_gap = 22
    left_w = 300
    center_w = 560
    right_w = 300
    left_x = inner_x
    center_x = left_x + left_w + col_gap
    right_x = center_x + center_w + col_gap
    prompt_h = 230
    photo_h = 410
    center_photo_h = 500
    meta_h = 220
    top_y = inner_y + 10
    bottom_y = top_y + prompt_h + 22

    def card(spec: tuple[int, int, int, int], *, fill: tuple[int, int, int, int] = palette["tile"], radius: int = 30, shadow: bool = True) -> None:
        if shadow:
            _draw_shadow(canvas, spec, radius, blur=14, offset=(0, 8), fill=(91, 71, 62, 22))
        draw.rounded_rectangle(spec, radius=radius, fill=fill, outline=palette["border"], width=2)

    def prompt_tile(spec: tuple[int, int, int, int], prompt: PromptBlock) -> None:
        card(spec)
        px0, py0, px1, py1 = spec
        draw.rounded_rectangle((px0 + 22, py0 + 22, px1 - 22, py0 + 68), radius=18, fill=palette["accent_soft"])
        prompt_text = _wrap_text(draw, prompt.prompt, prompt_font, px1 - px0 - 60, max_lines=2)
        response_text = _wrap_text(draw, prompt.response, body_font, px1 - px0 - 60, max_lines=3)
        draw.multiline_text((px0 + 30, py0 + 28), prompt_text, font=prompt_font, fill=palette["text"], spacing=6)
        draw.multiline_text((px0 + 30, py0 + 98), response_text, font=body_font, fill=palette["muted"], spacing=6)

    def image_tile(spec: tuple[int, int, int, int], image_path: str, *, label: str | None = None) -> None:
        card(spec)
        ix0, iy0, ix1, iy1 = spec
        inset = (ix0 + 18, iy0 + 18, ix1 - 18, iy1 - 18)
        _paste_cover_image(canvas, image_path, inset, 22)
        if label:
            pill_rect = (inset[0] + 18, inset[1] + 18, inset[0] + 140, inset[1] + 56)
            draw.rounded_rectangle(pill_rect, radius=19, fill=palette["pill"])
            bbox = draw.textbbox((0, 0), label, font=small_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (pill_rect[0] + (pill_rect[2] - pill_rect[0] - tw) / 2.0, pill_rect[1] + (pill_rect[3] - pill_rect[1] - th) / 2.0 - 1),
                label,
                font=small_font,
                fill=(255, 248, 245, 255),
            )

    def meta_tile(spec: tuple[int, int, int, int]) -> None:
        card(spec)
        mx0, my0, mx1, my1 = spec
        badge_rect = (mx0 + 30, my0 + 28, mx0 + 255, my0 + 74)
        draw.rounded_rectangle(badge_rect, radius=18, fill=palette["accent_soft"])
        badge_bbox = draw.textbbox((0, 0), "Featured Profile", font=badge_font)
        badge_w = badge_bbox[2] - badge_bbox[0]
        badge_h = badge_bbox[3] - badge_bbox[1]
        draw.text(
            (badge_rect[0] + (badge_rect[2] - badge_rect[0] - badge_w) / 2.0, badge_rect[1] + (badge_rect[3] - badge_rect[1] - badge_h) / 2.0 - 1),
            "Featured Profile",
            font=badge_font,
            fill=palette["accent"],
        )
        name_text = _wrap_text(draw, profile.name, title_font, mx1 - mx0 - 220, max_lines=2)
        draw.multiline_text((mx0 + 30, my0 + 90), name_text, font=title_font, fill=palette["text"], spacing=4)
        age_bbox = draw.textbbox((0, 0), str(profile.age), font=age_font)
        draw.text((mx1 - 30 - (age_bbox[2] - age_bbox[0]), my0 + 96), str(profile.age), font=age_font, fill=palette["accent"])
        occupation = _wrap_text(draw, profile.occupation, meta_font, mx1 - mx0 - 60, max_lines=2)
        draw.multiline_text((mx0 + 30, my0 + 178), occupation, font=meta_font, fill=palette["muted"], spacing=2)

    left_prompt_rect = (left_x, top_y, left_x + left_w, top_y + prompt_h)
    right_prompt_rect = (right_x, top_y, right_x + right_w, top_y + prompt_h)
    center_photo_rect = (center_x, top_y, center_x + center_w, top_y + center_photo_h)
    left_photo_rect = (left_x, bottom_y, left_x + left_w, bottom_y + photo_h)
    right_photo_rect = (right_x, bottom_y, right_x + right_w, bottom_y + photo_h)
    meta_rect = (center_x, top_y + center_photo_h + 32, center_x + center_w, top_y + center_photo_h + 32 + meta_h)

    prompt_tile(left_prompt_rect, profile.prompt_1)
    prompt_tile(right_prompt_rect, profile.prompt_2)
    image_tile(center_photo_rect, profile.picture_2, label="Top pick")
    image_tile(left_photo_rect, profile.picture_1, label="Photo 1")
    image_tile(right_photo_rect, profile.picture_3, label="Photo 3")
    meta_tile(meta_rect)

    return canvas.convert("RGB")


def _save_rendered_profile_image(profile: HingeProfile) -> Path:
    out_dir = Path(tempfile.gettempdir()) / "bci_hinge_profile_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{profile.folder_name}_profile.png"
    _render_profile_image(profile).save(out_path, format="PNG")
    return out_path


def _make_profile_stims(win: visual.Window, profile: HingeProfile, task_cfg: HingeTaskConfig) -> ProfileStimBundle:
    profile_img_path = _save_rendered_profile_image(profile)
    with Image.open(profile_img_path) as profile_img:
        img_w, img_h = profile_img.size
    aspect = float(img_w) / float(max(1, img_h))
    max_h = 0.78
    stim_h = max_h
    stim_w = stim_h * aspect
    max_w = float(win.size[0]) / float(max(1, win.size[1])) * 0.92
    if stim_w > max_w:
        stim_w = max_w
        stim_h = stim_w / aspect
    profile_stim = visual.ImageStim(
        win,
        image=str(profile_img_path),
        pos=(0.0, -0.045),
        size=(stim_w, stim_h),
        interpolate=True,
    )
    return ProfileStimBundle(stims=[profile_stim], temp_paths=[profile_img_path])


def _set_stim_x(stim: object, dx: float) -> None:
    pos = getattr(stim, "pos", None)
    if pos is not None:
        stim.pos = (float(pos[0]) + dx, float(pos[1]))
    vertices = getattr(stim, "vertices", None)
    if vertices is not None:
        shifted = []
        for vx, vy in vertices:
            shifted.append((float(vx) + dx, float(vy)))
        stim.vertices = shifted


def _set_alpha(bundle: ProfileStimBundle, alpha: float) -> None:
    for stim in bundle.stims:
        stim.opacity = alpha


def _draw_profile_bundle(bundle: ProfileStimBundle) -> None:
    for stim in bundle.stims:
        stim.draw()


def _create_hinge_window(task_cfg: HingeTaskConfig) -> tuple[visual.Window, dict[str, object]]:
    local_win = visual.Window(
        size=task_cfg.win_size,
        color=(0.985, 0.962, 0.950),
        units="height",
        fullscr=task_cfg.fullscreen,
    )
    chrome = {
        "app_badge": visual.Rect(
            local_win,
            width=0.13,
            height=0.028,
            pos=(0.0, 0.485),
            fillColor=(0.965, 0.905, 0.915),
            lineColor=None,
        ),
        "app_label": visual.TextStim(
            local_win,
            text="PROFILE",
            pos=(0.0, 0.485),
            height=0.015,
            color=(-0.30, -0.16, -0.02),
            bold=True,
        ),
        "command_shadow": visual.Rect(
            local_win,
            width=1.18,
            height=0.072,
            pos=(0.0, 0.428),
            fillColor=(0.91, 0.87, 0.85),
            lineColor=None,
            opacity=0.24,
        ),
        "command_box": visual.Rect(
            local_win,
            width=1.16,
            height=0.068,
            pos=(0.0, 0.432),
            fillColor=(0.995, 0.992, 0.989),
            lineColor=(0.90, 0.86, 0.84),
            lineWidth=1.8,
        ),
        "command_text": visual.TextStim(
            local_win,
            text="Jaw clench when you are ready to choose.",
            pos=(0.0, 0.444),
            height=0.022,
            color=(-0.34, -0.22, -0.08),
            bold=True,
            wrapWidth=1.06,
        ),
        "sub_text": visual.TextStim(
            local_win,
            text="",
            pos=(0.0, 0.420),
            height=0.013,
            color=(-0.55, -0.47, -0.36),
            wrapWidth=1.08,
        ),
        "footer": visual.TextStim(
            local_win,
            text="Jaw clench to start a swipe trial. ESC quits.",
            pos=(0.0, -0.485),
            height=0.014,
            color=(-0.44, -0.34, -0.22),
            wrapWidth=1.15,
        ),
        "heart": visual.TextStim(
            local_win,
            text="❤",
            pos=(0.0, 0.0),
            height=0.18,
            color=(0.92, -0.05, 0.12),
            opacity=0.0,
        ),
        "cross_bg": visual.Circle(
            local_win,
            radius=0.10,
            pos=(0.0, 0.0),
            fillColor=(0.96, 0.96, 0.96),
            lineColor=None,
            opacity=0.0,
        ),
        "cross": visual.TextStim(
            local_win,
            text="✕",
            pos=(0.0, 0.0),
            height=0.14,
            color=(-1.0, -1.0, -1.0),
            opacity=0.0,
        ),
    }
    return local_win, chrome


def preview_profiles(profile_name: str | None = None) -> None:
    task_cfg = HingeTaskConfig()
    cfgs = apply_runtime_config_overrides("hinge_task", task_cfg=task_cfg)
    task_cfg = cfgs["task_cfg"]

    profiles = _load_profiles(task_cfg.profiles_dir)
    if profile_name is not None:
        profile_name_key = str(profile_name).strip().lower()
        profiles = [p for p in profiles if p.folder_name.lower() == profile_name_key]
        if not profiles:
            raise RuntimeError(f"No profile folder named {profile_name!r} found under {task_cfg.profiles_dir}")

    win, chrome = _create_hinge_window(task_cfg)
    profile_index = 0

    try:
        while True:
            profile = profiles[profile_index]
            bundle = _make_profile_stims(win, profile, task_cfg)
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    return
                if "right" in keys and len(profiles) > 1:
                    profile_index = (profile_index + 1) % len(profiles)
                    break
                if "left" in keys and len(profiles) > 1:
                    profile_index = (profile_index - 1) % len(profiles)
                    break

                chrome["command_text"].text = "PROFILE PREVIEW"
                chrome["sub_text"].text = (
                    f"{profile.folder_name}   "
                    + ("LEFT/RIGHT arrows switch profiles. " if len(profiles) > 1 else "")
                    + "ESC closes preview."
                )
                chrome["footer"].text = "Preview mode only. No EEG, jaw clench, or classification."
                _draw_profile_bundle(bundle)
                chrome["command_shadow"].draw()
                chrome["command_box"].draw()
                chrome["app_badge"].draw()
                chrome["app_label"].draw()
                chrome["command_text"].draw()
                chrome["sub_text"].draw()
                chrome["footer"].draw()
                win.flip()
    finally:
        _close_window(win)


def _open_stream(lsl_cfg: LSLConfig, eeg_cfg: EEGConfig) -> tuple[StreamLSL, float, list[str], list[str]]:
    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    available = list(stream.info["ch_names"])
    model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    if len(model_ch_names) < 2:
        event_key = canonicalize_channel_name(lsl_cfg.event_channels)
        model_ch_names = [ch for ch in available if canonicalize_channel_name(ch) != event_key]
    if len(model_ch_names) < 2:
        raise RuntimeError(f"Need >=2 EEG channels, found: {available}")
    stream.pick(model_ch_names)
    sfreq = float(stream.info["sfreq"])
    return stream, sfreq, model_ch_names, missing


def _close_stream(stream: StreamLSL | None) -> None:
    if stream is None:
        return
    try:
        stream.disconnect()
    except Exception:
        pass


def _close_window(win: visual.Window | None) -> None:
    if win is None:
        return
    try:
        win.close()
    except Exception:
        pass


def run_task(fname: str, max_trials: int | None = None) -> None:  # noqa: C901
    logger = _make_task_logger(fname)

    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    task_cfg = HingeTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )
    cfgs = apply_runtime_config_overrides(
        "hinge_task",
        lsl_cfg=lsl_cfg,
        stim_cfg=stim_cfg,
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        eeg_cfg=eeg_cfg,
    )
    lsl_cfg = cfgs["lsl_cfg"]
    stim_cfg = cfgs["stim_cfg"]
    task_cfg = cfgs["task_cfg"]
    model_cfg = cfgs["model_cfg"]
    eeg_cfg = cfgs["eeg_cfg"]

    profiles = _load_profiles(task_cfg.profiles_dir)
    rng = random.Random()
    rng.shuffle(profiles)
    logger.info("Loaded %d hinge profiles from %s", len(profiles), task_cfg.profiles_dir)

    stream: StreamLSL | None = None
    win: visual.Window | None = None

    def _draw(bundle: ProfileStimBundle, chrome: dict[str, object], *, command: str, sub: str = "", dx: float = 0.0, alpha: float = 1.0, overlay: str | None = None, overlay_alpha: float = 0.0) -> None:
        local_win = chrome["command_box"].win
        local_win.color = (0.985, 0.962, 0.950)
        chrome["command_text"].text = command
        chrome["sub_text"].text = sub

        # Rebuild the translated positions each frame from the canonical objects.
        _set_alpha(bundle, alpha)
        if dx != 0.0:
            for stim in bundle.stims:
                _set_stim_x(stim, dx)

        _draw_profile_bundle(bundle)
        chrome["command_shadow"].draw()
        chrome["command_box"].draw()
        chrome["app_badge"].draw()
        chrome["app_label"].draw()
        chrome["command_text"].draw()
        chrome["sub_text"].draw()
        chrome["footer"].draw()
        if overlay == "right":
            chrome["heart"].opacity = overlay_alpha
            chrome["heart"].draw()
        elif overlay == "left":
            chrome["cross_bg"].opacity = overlay_alpha
            chrome["cross"].opacity = overlay_alpha
            chrome["cross_bg"].draw()
            chrome["cross"].draw()
        local_win.flip()
        if dx != 0.0:
            for stim in bundle.stims:
                _set_stim_x(stim, -dx)

    try:
        stream, sfreq, model_ch_names, missing = _open_stream(lsl_cfg, eeg_cfg)
        logger.info("Connected hinge stream: sfreq=%.3f selected=%s missing=%s", sfreq, model_ch_names, missing)

        shared_epoch_model = resolve_shared_epoch_mi_model(
            cache_name="mi_shared_epoch_model",
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            calibrate_on_participant=task_cfg.calirate_on_participant,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            model_cfg=model_cfg,
            target_sfreq=float(sfreq),
            target_channel_names=model_ch_names,
            logger=logger,
        )
        epoch_classifier = shared_epoch_model.classifier
        epoch_class_index = shared_epoch_model.class_index
        if int(stim_cfg.left_code) not in epoch_class_index or int(stim_cfg.right_code) not in epoch_class_index:
            raise RuntimeError("Shared epoch MI model does not contain expected left/right classes.")
        with open(f"{fname}_hinge_epoch_model.pkl", "wb") as fh:
            pickle.dump(epoch_classifier, fh)

        win, chrome = _create_hinge_window(task_cfg)
        jaw_idxs = select_jaw_channel_indices(model_ch_names)
        jaw_window_n = int(round(float(task_cfg.jaw_window_s) * float(sfreq)))
        jaw_classifier = None
        runtime_jaw_classifier, runtime_train_acc = resolve_runtime_jaw_classifier(logger=logger, min_total_samples=12)
        if runtime_jaw_classifier is not None:
            jaw_classifier = runtime_jaw_classifier
            logger.info("Using orchestrator-provided jaw calibration (train_acc=%.3f).", float(runtime_train_acc or 0.0))
        else:
            def _wait_for_space(prompt_text: str) -> None:
                while True:
                    chrome["command_text"].text = prompt_text
                    chrome["sub_text"].text = "Press SPACE to continue. ESC quits."
                    chrome["command_shadow"].draw()
                    chrome["command_box"].draw()
                    chrome["app_badge"].draw()
                    chrome["app_label"].draw()
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    win.flip()
                    keys = event.getKeys()
                    if "escape" in keys:
                        raise KeyboardInterrupt
                    if "space" in keys:
                        return

            def _wait_for_seconds(duration_s: float) -> None:
                clk = core.Clock()
                while clk.getTime() < duration_s:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    chrome["command_shadow"].draw()
                    chrome["command_box"].draw()
                    chrome["app_badge"].draw()
                    chrome["app_label"].draw()
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    win.flip()

            def _collect_stream_block(duration_s: float) -> np.ndarray:
                chunks: list[np.ndarray] = []
                last_ts_local: float | None = None
                clk = core.Clock()
                while clk.getTime() < duration_s:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    data, ts = stream.get_data(winsize=min(0.20, duration_s), picks="all")
                    if data.size > 0 and ts is not None and len(ts) > 0:
                        ts_arr = np.asarray(ts)
                        mask = np.ones_like(ts_arr, dtype=bool) if last_ts_local is None else (ts_arr > float(last_ts_local))
                        if np.any(mask):
                            chunks.append(np.asarray(data[:, mask], dtype=np.float32))
                            last_ts_local = float(ts_arr[mask][-1])
                    chrome["command_shadow"].draw()
                    chrome["command_box"].draw()
                    chrome["app_badge"].draw()
                    chrome["app_label"].draw()
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    win.flip()
                if not chunks:
                    return np.empty((len(model_ch_names), 0), dtype=np.float32)
                return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

            jaw_classifier, jaw_train_acc, _jaw_y = run_visual_jaw_calibration(
                cue=chrome["command_text"],
                info=chrome["sub_text"],
                status=chrome["footer"],
                wait_for_space=_wait_for_space,
                wait_for_seconds=_wait_for_seconds,
                collect_stream_block=_collect_stream_block,
                jaw_idxs=jaw_idxs,
                jaw_window_n=jaw_window_n,
                sfreq=sfreq,
                model_ch_names=model_ch_names,
                logger=logger,
                n_per_class=int(task_cfg.jaw_calibration_blocks_per_class),
                hold_s=float(task_cfg.jaw_calibration_hold_s),
                prep_s=float(task_cfg.jaw_calibration_prep_s),
                iti_s=float(task_cfg.jaw_calibration_iti_s),
                window_s=float(task_cfg.jaw_window_s),
                step_s=float(task_cfg.jaw_window_step_s),
                edge_trim_s=float(task_cfg.jaw_calibration_trim_s),
                min_total_samples=12,
            )
            logger.info("Runtime jaw calibration complete: train_acc=%.3f", float(jaw_train_acc))

        epoch_n = int(round(float(task_cfg.epoch_duration_s) * float(sfreq)))
        context_n = int(round(float(task_cfg.filter_context_s) * float(sfreq)))
        keep_n = max(epoch_n + context_n, jaw_window_n + 1)
        reject_thresh = eeg_cfg.reject_peak_to_peak
        trial_results: list[dict[str, object]] = []
        profile_cycle = [profiles[i % len(profiles)] for i in range(int(max_trials or len(profiles)))]
        if max_trials is not None and len(profile_cycle) < int(max_trials):
            while len(profile_cycle) < int(max_trials):
                cycle_copy = profiles[:]
                rng.shuffle(cycle_copy)
                profile_cycle.extend(cycle_copy)
            profile_cycle = profile_cycle[: int(max_trials)]

        for trial_idx, profile in enumerate(profile_cycle, start=1):
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

            profile_bundle = _make_profile_stims(win, profile, task_cfg)
            raw_history = np.empty((len(model_ch_names), 0), dtype=np.float32)
            jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
            jaw_prob = 0.0
            jaw_prev_pred = 0
            jaw_event_pending = False
            jaw_last_trigger_t = -1e9
            last_live_ts: float | None = None

            def _poll_live() -> None:
                nonlocal last_live_ts, raw_history, jaw_buffer, jaw_prob, jaw_prev_pred, jaw_event_pending, jaw_last_trigger_t
                data, ts = stream.get_data(winsize=0.20, picks="all")
                if data.size == 0 or ts is None or len(ts) == 0:
                    return
                ts_arr = np.asarray(ts)
                mask = np.ones_like(ts_arr, dtype=bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
                if not np.any(mask):
                    return
                x_new = np.asarray(data[:, mask], dtype=np.float32)
                last_live_ts = float(ts_arr[mask][-1])
                raw_history = np.concatenate((raw_history, x_new), axis=1)
                if raw_history.shape[1] > keep_n:
                    raw_history = raw_history[:, -keep_n:]
                jaw_buffer, jaw_prob, jaw_prev_pred, should_toggle = update_live_jaw_clench_state(
                    jaw_buffer=jaw_buffer,
                    x_new=x_new,
                    keep_n=keep_n,
                    jaw_window_n=jaw_window_n,
                    jaw_classifier=jaw_classifier,
                    jaw_idxs=jaw_idxs,
                    jaw_prob=jaw_prob,
                    jaw_prev_pred=jaw_prev_pred,
                    jaw_prob_thresh=float(task_cfg.jaw_clench_prob_thresh),
                    jaw_last_toggle_t=jaw_last_trigger_t,
                    jaw_refractory_s=float(task_cfg.jaw_clench_refractory_s),
                    now_t=core.getTime(),
                )
                if should_toggle:
                    jaw_last_trigger_t = core.getTime()
                    jaw_event_pending = True

            def _collect_execute_block(duration_s: float) -> np.ndarray:
                chunks: list[np.ndarray] = []
                local_last_ts = last_live_ts
                clk = core.Clock()
                while clk.getTime() < duration_s:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    data, ts = stream.get_data(winsize=min(0.20, duration_s), picks="all")
                    if data.size > 0 and ts is not None and len(ts) > 0:
                        ts_arr = np.asarray(ts)
                        mask = np.ones_like(ts_arr, dtype=bool) if local_last_ts is None else (ts_arr > float(local_last_ts))
                        if np.any(mask):
                            x_new = np.asarray(data[:, mask], dtype=np.float32)
                            chunks.append(x_new)
                            raw_history_local = np.concatenate((raw_history, x_new), axis=1)
                            local_last_ts = float(ts_arr[mask][-1])
                            _ = raw_history_local  # keeps parity with the live path for readability
                    _draw(
                        profile_bundle,
                        chrome,
                        command="EXECUTE SWIPE",
                        sub=f"Perform your chosen imagery now. {max(0.0, duration_s - clk.getTime()):.1f}s remaining",
                    )
                nonlocal_last_ts_holder[0] = local_last_ts
                if not chunks:
                    return np.empty((len(model_ch_names), 0), dtype=np.float32)
                return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

            nonlocal_last_ts_holder = [last_live_ts]
            idle_text = "Jaw clench when you are ready to swipe."
            while True:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                _poll_live()
                _draw(
                    profile_bundle,
                    chrome,
                    command=idle_text,
                    sub=f"Trial {trial_idx}/{len(profile_cycle)}   jaw_p={jaw_prob:.2f}   Decide left to pass or right to like.",
                )
                if jaw_event_pending:
                    jaw_event_pending = False
                    break

            prep_clock = core.Clock()
            while prep_clock.getTime() < float(task_cfg.prep_duration_s):
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                _poll_live()
                remaining = max(0.0, float(task_cfg.prep_duration_s) - prep_clock.getTime())
                _draw(
                    profile_bundle,
                    chrome,
                    command="PREPARE TO SWIPE LEFT TO PASS OR RIGHT TO LIKE",
                    sub=f"Prepare your chosen imagery. {remaining:.1f}s remaining",
                )

            pre_context = raw_history[:, -context_n:].copy() if context_n > 0 and raw_history.shape[1] > 0 else np.empty((len(model_ch_names), 0), dtype=np.float32)
            _draw(profile_bundle, chrome, command="EXECUTE SWIPE", sub="Perform your chosen imagery now.")
            execute_flip_time = core.getTime()
            _ = execute_flip_time
            execute_block_chunks: list[np.ndarray] = []
            local_last_ts = last_live_ts
            execute_clock = core.Clock()
            while execute_clock.getTime() < float(task_cfg.execute_duration_s):
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                data, ts = stream.get_data(winsize=min(0.20, float(task_cfg.execute_duration_s)), picks="all")
                if data.size > 0 and ts is not None and len(ts) > 0:
                    ts_arr = np.asarray(ts)
                    mask = np.ones_like(ts_arr, dtype=bool) if local_last_ts is None else (ts_arr > float(local_last_ts))
                    if np.any(mask):
                        x_new = np.asarray(data[:, mask], dtype=np.float32)
                        execute_block_chunks.append(x_new)
                        local_last_ts = float(ts_arr[mask][-1])
                remaining = max(0.0, float(task_cfg.execute_duration_s) - execute_clock.getTime())
                _draw(
                    profile_bundle,
                    chrome,
                    command="EXECUTE SWIPE",
                    sub=f"Perform your chosen imagery now. {remaining:.1f}s remaining",
                )
            last_live_ts = local_last_ts
            execute_block = (
                np.concatenate(execute_block_chunks, axis=1).astype(np.float32, copy=False)
                if execute_block_chunks
                else np.empty((len(model_ch_names), 0), dtype=np.float32)
            )
            raw_history = np.concatenate((raw_history, execute_block), axis=1)
            if raw_history.shape[1] > keep_n:
                raw_history = raw_history[:, -keep_n:]

            _draw(profile_bundle, chrome, command="STOP", sub="Decoding your swipe...")
            if execute_block.shape[1] < epoch_n:
                raise RuntimeError(
                    f"Collected only {execute_block.shape[1]} samples during execute, need at least {epoch_n}."
                )
            classify_block = np.concatenate((pre_context, execute_block), axis=1) if pre_context.size else execute_block
            classify_filt = filter_session(classify_block, eeg_cfg=eeg_cfg, sfreq=float(sfreq))
            epoch_block = classify_filt[:, -epoch_n:]
            max_ptp = float(np.ptp(epoch_block, axis=-1).max())
            if reject_thresh is not None and max_ptp > float(reject_thresh):
                logger.warning(
                    "Hinge trial %d exceeded artifact threshold but will still be classified: max_ptp=%.3f thresh=%.3f",
                    trial_idx,
                    max_ptp,
                    float(reject_thresh),
                )

            p_vec = epoch_classifier.predict_proba(epoch_block[np.newaxis, ...])[0]
            left_prob = float(p_vec[epoch_class_index[int(stim_cfg.left_code)]])
            right_prob = float(p_vec[epoch_class_index[int(stim_cfg.right_code)]])
            pred_code = int(stim_cfg.right_code) if right_prob >= left_prob else int(stim_cfg.left_code)
            swipe_direction = "right" if pred_code == int(stim_cfg.right_code) else "left"

            swipe_clock = core.Clock()
            swipe_duration = 0.42
            while swipe_clock.getTime() < swipe_duration:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                progress = min(1.0, swipe_clock.getTime() / swipe_duration)
                eased = 1.0 - (1.0 - progress) ** 3
                dx = (1.55 * eased) * (1.0 if swipe_direction == "right" else -1.0)
                alpha = max(0.0, 1.0 - progress * 0.80)
                _draw(
                    profile_bundle,
                    chrome,
                    command="STOP",
                    sub=f"Swipe decoded: {'LIKE' if swipe_direction == 'right' else 'PASS'}",
                    dx=dx,
                    alpha=alpha,
                )

            overlay_clock = core.Clock()
            while overlay_clock.getTime() < float(task_cfg.outcome_duration_s):
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                progress = min(1.0, overlay_clock.getTime() / float(task_cfg.outcome_duration_s))
                overlay_alpha = 1.0 - 0.45 * progress
                _draw(
                    profile_bundle,
                    chrome,
                    command="MATCH!" if swipe_direction == "right" else "PASS",
                    sub=(
                        f"Decoded RIGHT  p_right={right_prob:.2f} p_left={left_prob:.2f}"
                        if swipe_direction == "right"
                        else f"Decoded LEFT  p_left={left_prob:.2f} p_right={right_prob:.2f}"
                    ),
                    overlay=swipe_direction,
                    overlay_alpha=overlay_alpha,
                )

            submitted_message: str | None = None
            if swipe_direction == "right" and bool(task_cfg.enable_keyboard_on_match):
                logger.info("Launching nested keyboard after right swipe for profile=%s", profile.folder_name)
                _close_stream(stream)
                stream = None
                _close_window(win)
                win = None
                submitted_message = run_keyboard_task(
                    fname=f"{fname}_match_{trial_idx:02d}_{profile.folder_name}",
                    move_confidence_thresh=float(task_cfg.keyboard_move_confidence_thresh),
                    cursor_step_s=float(task_cfg.keyboard_cursor_step_s),
                    jaw_select_refractory_s=float(task_cfg.keyboard_jaw_select_refractory_s),
                    max_text_chars=int(task_cfg.keyboard_max_text_chars),
                    raise_on_escape=True,
                )
                stream, sfreq, model_ch_names, missing = _open_stream(lsl_cfg, eeg_cfg)
                jaw_idxs = select_jaw_channel_indices(model_ch_names)
                jaw_window_n = int(round(float(task_cfg.jaw_window_s) * float(sfreq)))
                logger.info("Reconnected hinge stream after keyboard: sfreq=%.3f selected=%s missing=%s", sfreq, model_ch_names, missing)
                win, chrome = _create_hinge_window(task_cfg)

            trial_result = {
                "trial": int(trial_idx),
                "profile_folder": profile.folder_name,
                "profile_name": profile.name,
                "left_prob": left_prob,
                "right_prob": right_prob,
                "pred_code": pred_code,
                "swipe_direction": swipe_direction,
                "artifact_ptp": max_ptp,
                "message_text": submitted_message,
            }
            trial_results.append(trial_result)
            logger.info("Hinge trial complete: %s", trial_result)

            pause_clock = core.Clock()
            while pause_clock.getTime() < float(task_cfg.inter_trial_pause_s):
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                if win is not None:
                    chrome["command_shadow"].draw()
                    chrome["command_box"].draw()
                    chrome["app_badge"].draw()
                    chrome["app_label"].draw()
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    win.flip()

        with open(f"{fname}_hinge_trials.pkl", "wb") as fh:
            pickle.dump(trial_results, fh)
        logger.info("Saved %d hinge trials to %s_hinge_trials.pkl", len(trial_results), fname)

    except KeyboardInterrupt:
        logger.info("Hinge task interrupted by user.")
    finally:
        _close_stream(stream)
        _close_window(win)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hinge-style BCI task")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the Hinge profile GUI without EEG, models, or jaw clench flow.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile folder name to preview under bci/profiles.",
    )
    args = parser.parse_args()

    if bool(args.preview):
        preview_profiles(profile_name=args.profile)
    else:
        prefix = _prompt_prefix()
        print(f"[SESSION] prefix: {prefix}")
        run_task(fname=prefix)
