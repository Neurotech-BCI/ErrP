from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from mne_lsl.stream import StreamLSL
from PIL import Image, ImageOps
from psychopy import core, event, visual

from bci_runtime import (
    apply_runtime_config_overrides,
    resolve_runtime_jaw_classifier,
    resolve_shared_epoch_mi_model,
)
from config import EEGConfig, HingeTaskConfig, LSLConfig, MentalCommandModelConfig, StimConfig
from derick_ml_jawclench import (
    collect_cue_locked_stream_block,
    run_visual_jaw_calibration,
    select_jaw_channel_indices,
    update_live_jaw_clench_state,
)
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
    dating_intentions: str
    prompt_1: PromptBlock
    prompt_2: PromptBlock
    picture_1: str
    picture_2: str
    picture_3: str


def _import_pygame():
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - depends on target machine environment
        raise RuntimeError(
            "pygame is required for hinge_task UI. Install it on the target machine with `pip install pygame`."
        ) from exc
    return pygame


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


def _truncate_chars(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 3)].rstrip() + "..."


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
                dating_intentions=str(meta.get("dating_intentions", "")),
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


class HongePygameUI:
    def __init__(self, task_cfg: HingeTaskConfig) -> None:
        pygame = _import_pygame()
        pygame.init()
        pygame.font.init()
        self.pygame = pygame
        self.win_size = (int(task_cfg.win_size[0]), int(task_cfg.win_size[1]))
        self.screen = pygame.display.set_mode(self.win_size)
        pygame.display.set_caption("Honge")
        self.clock = pygame.time.Clock()
        self.closed = False
        self._keys: list[str] = []

        self.scale = min(self.win_size[0] / 1280.0, self.win_size[1] / 760.0)
        self.palette = {
            "bg_top": (250, 238, 233),
            "bg_bottom": (246, 239, 234),
            "board": (255, 252, 250),
            "board_line": (233, 223, 216),
            "card": (255, 255, 255),
            "card_alt": (252, 248, 245),
            "card_line": (232, 223, 216),
            "card_shadow": (124, 94, 81),
            "text": (30, 24, 22),
            "muted": (111, 97, 90),
            "soft_muted": (147, 132, 123),
            "accent": (214, 96, 74),
            "accent_soft": (247, 225, 219),
            "accent_wash": (255, 244, 239),
            "hero_overlay": (19, 16, 17),
            "white": (255, 255, 255),
            "rule": (237, 229, 223),
        }

        self.board_radius = self._s(38)
        self.card_radius = self._s(28)
        self.media_radius = self._s(26)

        self.logo_font = self._font(24, bold=True)
        self.status_title_font = self._font(21, bold=True)
        self.status_body_font = self._font(15)
        self.footer_font = self._font(14)
        self.prompt_kicker_font = self._font(16)
        self.table_label_font = self._font(13, bold=True)
        self.table_value_font = self._font(15)
        self.table_value_emphasis_font = self._font(16, bold=True)
        self.overlay_font = self._font(42, bold=True)

        self.brand_rect = self._rect(524, 18, 232, 56)
        self.status_rect = self._rect(272, 86, 736, 62)
        self.footer_rect = self._rect(240, 720, 800, 20)
        self.profile_rect = self._rect(109, 150, 1062, 548)

        pad = self._s(22)
        gap = self._s(18)
        narrow_w = self._s(248)
        wide_w = self._s(486)
        prompt_h = self._s(176)
        hero_h = self._s(322)
        info_h = self._s(164)
        side_photo_h = self._s(310)

        left_x = self.profile_rect.x + pad
        center_x = left_x + narrow_w + gap
        right_x = center_x + wide_w + gap
        top_y = self.profile_rect.y + pad
        bottom_y = top_y + prompt_h + gap
        info_y = top_y + hero_h + gap

        self.left_prompt_rect = pygame.Rect(left_x, top_y, narrow_w, prompt_h)
        self.center_photo_rect = pygame.Rect(center_x, top_y, wide_w, hero_h)
        self.right_prompt_rect = pygame.Rect(right_x, top_y, narrow_w, prompt_h)
        self.left_photo_rect = pygame.Rect(left_x, bottom_y, narrow_w, side_photo_h)
        self.meta_rect = pygame.Rect(center_x, info_y, wide_w, info_h)
        self.right_photo_rect = pygame.Rect(right_x, bottom_y, narrow_w, side_photo_h)

        self.command_text = ""
        self.sub_text = ""
        self.footer_text = "Jaw clench to start a swipe trial. ESC quits."
        self.profile_surface: object | None = None
        self.profile_surface_shadow: object | None = None
        self.profile_offset_x = 0
        self.profile_opacity = 255
        self.profile_visible = True
        self.overlay_kind: str | None = None
        self.overlay_alpha = 0
        self.background_surface = self._build_background_surface()

    def _s(self, value: int | float) -> int:
        return max(1, int(round(float(value) * self.scale)))

    def _rect(self, x: int, y: int, w: int, h: int):
        return self.pygame.Rect(self._s(x), self._s(y), self._s(w), self._s(h))

    def _font(self, px: int, *, bold: bool = False, serif: bool = False, italic: bool = False):
        pygame = self.pygame
        families = (
            ["Iowan Old Style", "Georgia", "Palatino Linotype", "Times New Roman", "DejaVu Serif"]
            if serif
            else ["Avenir Next", "Aptos", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
        )
        for name in families:
            path = pygame.font.match_font(name, bold=bold, italic=italic)
            if path:
                return pygame.font.Font(path, px)
        return pygame.font.SysFont(None, px, bold=bold, italic=italic)

    def _rounded(self, surf, rect, color, *, radius: int, border: int = 0, border_color=None) -> None:
        self.pygame.draw.rect(surf, color, rect, border_radius=radius)
        if border > 0 and border_color is not None:
            self.pygame.draw.rect(surf, border_color, rect, width=border, border_radius=radius)

    def _build_background_surface(self):
        pygame = self.pygame
        w, h = self.win_size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        top = self.palette["bg_top"]
        bottom = self.palette["bg_bottom"]
        for y in range(h):
            t = y / max(1, h - 1)
            color = tuple(int(round(top[i] * (1.0 - t) + bottom[i] * t)) for i in range(3))
            pygame.draw.line(surf, color, (0, y), (w, y))

        accents = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(
            accents,
            (*self.palette["accent_soft"], 115),
            pygame.Rect(self._s(-40), self._s(-32), self._s(300), self._s(220)),
        )
        pygame.draw.ellipse(
            accents,
            (255, 255, 255, 135),
            pygame.Rect(self._s(960), self._s(70), self._s(260), self._s(180)),
        )
        pygame.draw.ellipse(
            accents,
            (255, 255, 255, 115),
            pygame.Rect(self._s(118), self._s(540), self._s(340), self._s(160)),
        )
        pygame.draw.ellipse(
            accents,
            (*self.palette["accent_soft"], 72),
            pygame.Rect(self._s(930), self._s(560), self._s(280), self._s(140)),
        )
        surf.blit(accents, (0, 0))
        return surf

    def _blit_lines(
        self,
        surf,
        font,
        lines: list[str],
        color,
        topleft: tuple[int, int],
        *,
        line_gap: int = 4,
    ) -> None:
        x, y = topleft
        line_h = font.get_height()
        for idx, line in enumerate(lines):
            img = font.render(line, True, color)
            surf.blit(img, (x, y + idx * (line_h + line_gap)))

    def _ellipsize(self, font, text: str, max_width: int) -> str:
        cleaned = " ".join(str(text).split())
        if font.size(cleaned)[0] <= max_width:
            return cleaned
        candidate = cleaned.rstrip(". ")
        while candidate and font.size(candidate + "...")[0] > max_width:
            candidate = candidate[:-1].rstrip()
        return (candidate or cleaned[:1]).rstrip() + "..."

    def _wrap_lines(self, font, text: str, max_width: int, *, max_lines: int | None = None) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]
        lines = [words[0]]
        for word in words[1:]:
            test = f"{lines[-1]} {word}"
            if font.size(test)[0] <= max_width:
                lines[-1] = test
            else:
                lines.append(word)
        if max_lines is not None and len(lines) > max_lines:
            overflow = " ".join(lines[max_lines - 1 :])
            lines = lines[: max_lines - 1] + [self._ellipsize(font, overflow, max_width)]
        return lines

    def _fit_wrapped_text(
        self,
        text: str,
        *,
        max_width: int,
        max_height: int,
        max_px: int,
        min_px: int,
        max_lines: int,
        bold: bool = False,
        serif: bool = False,
        line_gap: int = 4,
    ):
        best_font = self._font(min_px, bold=bold, serif=serif)
        best_lines = self._wrap_lines(best_font, text, max_width, max_lines=max_lines)
        for px in range(max_px, min_px - 1, -2):
            font = self._font(px, bold=bold, serif=serif)
            lines = self._wrap_lines(font, text, max_width, max_lines=max_lines)
            total_h = len(lines) * font.get_height() + max(0, len(lines) - 1) * line_gap
            if total_h <= max_height:
                return font, lines
            best_font, best_lines = font, lines
        return best_font, best_lines

    def _load_cover(self, image_path: str, size: tuple[int, int]):
        pygame = self.pygame
        target_w, target_h = int(size[0]), int(size[1])
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        surf = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode).convert()
        scale = max(target_w / surf.get_width(), target_h / surf.get_height())
        scaled = pygame.transform.smoothscale(
            surf,
            (max(1, int(round(surf.get_width() * scale))), max(1, int(round(surf.get_height() * scale)))),
        )
        left = max(0, int((scaled.get_width() - target_w) / 2))
        top = max(0, int((scaled.get_height() - target_h) / 2))
        return scaled.subsurface((left, top, target_w, target_h)).copy()

    def _clip_round(self, surf, *, radius: int):
        mask = self.pygame.Surface(surf.get_size(), self.pygame.SRCALPHA)
        self._rounded(mask, mask.get_rect(), (255, 255, 255, 255), radius=radius)
        clipped = self.pygame.Surface(surf.get_size(), self.pygame.SRCALPHA)
        clipped.blit(surf, (0, 0))
        clipped.blit(mask, (0, 0), special_flags=self.pygame.BLEND_RGBA_MULT)
        return clipped

    def _load_rounded_cover(self, image_path: str, size: tuple[int, int], *, radius: int):
        return self._clip_round(self._load_cover(image_path, size), radius=radius)

    def _make_shadow_surface(self, size: tuple[int, int], *, radius: int, layers: list[tuple[int, int, int]]):
        padding = self._s(26)
        surf = self.pygame.Surface((size[0] + padding * 2, size[1] + padding * 2 + self._s(14)), self.pygame.SRCALPHA)
        base_rect = self.pygame.Rect(padding, padding, size[0], size[1])
        for spread, alpha, dy in layers:
            shadow_rect = base_rect.inflate(self._s(spread), self._s(spread)).move(0, self._s(dy))
            self._rounded(
                surf,
                shadow_rect,
                (*self.palette["card_shadow"], alpha),
                radius=radius + self._s(max(2, spread // 3)),
            )
        return surf

    def _draw_card_shell(self, surf, rect, *, fill_color, radius: int | None = None) -> None:
        radius = self.card_radius if radius is None else radius
        shadow = self._make_shadow_surface(
            rect.size,
            radius=radius,
            layers=[(24, 10, 4), (10, 18, 10)],
        )
        shadow_pad_x = (shadow.get_width() - rect.width) // 2
        shadow_pad_y = max(0, (shadow.get_height() - rect.height) // 2 - self._s(5))
        surf.blit(shadow, (rect.x - shadow_pad_x, rect.y - shadow_pad_y))
        self._rounded(surf, rect, fill_color, radius=radius, border=1, border_color=self.palette["card_line"])

    def _draw_centered_lines(self, surf, font, lines: list[str], color, rect, *, line_gap: int = 4) -> None:
        total_h = len(lines) * font.get_height() + max(0, len(lines) - 1) * line_gap
        y = rect.y + (rect.height - total_h) // 2
        for line in lines:
            img = font.render(line, True, color)
            surf.blit(img, (rect.centerx - img.get_width() // 2, y))
            y += font.get_height() + line_gap

    def _draw_hero_text(self, surf, profile: HingeProfile, rect) -> None:
        overlay = self.pygame.Surface(rect.size, self.pygame.SRCALPHA)
        for y in range(rect.height):
            t = y / max(1, rect.height - 1)
            alpha = int(220 * (max(0.0, t - 0.35) / 0.65) ** 1.65)
            self.pygame.draw.line(
                overlay,
                (*self.palette["hero_overlay"], alpha),
                (0, y),
                (rect.width, y),
            )
        surf.blit(overlay, rect.topleft)

        pip_w = self._s(38)
        pip_h = self._s(5)
        pip_gap = self._s(8)
        total_w = pip_w * 3 + pip_gap * 2
        pip_x = rect.centerx - total_w // 2
        pip_y = rect.y + self._s(18)
        for idx in range(3):
            fill = (255, 255, 255, 240 if idx == 1 else 120)
            pip = self.pygame.Surface((pip_w, pip_h), self.pygame.SRCALPHA)
            self._rounded(pip, pip.get_rect(), fill, radius=max(2, pip_h // 2))
            surf.blit(pip, (pip_x + idx * (pip_w + pip_gap), pip_y))

    def _draw_prompt_card(self, surf, rect, prompt: PromptBlock) -> None:
        self._draw_card_shell(surf, rect, fill_color=self.palette["card"])
        kicker = self.prompt_kicker_font.render(_truncate_chars(prompt.prompt, 48), True, self.palette["muted"])
        surf.blit(kicker, (rect.x + self._s(22), rect.y + self._s(20)))

        body_rect = self.pygame.Rect(
            rect.x + self._s(22),
            rect.y + self._s(52),
            rect.width - self._s(44),
            rect.height - self._s(72),
        )
        font, lines = self._fit_wrapped_text(
            prompt.response,
            max_width=body_rect.width,
            max_height=body_rect.height,
            max_px=self._s(34),
            min_px=self._s(21),
            max_lines=4,
            bold=True,
            serif=True,
            line_gap=self._s(3),
        )
        self._blit_lines(
            surf,
            font,
            lines,
            self.palette["text"],
            (body_rect.x, body_rect.y),
            line_gap=self._s(3),
        )

    def _draw_media_card(self, surf, rect, image_path: str, *, hero: bool = False) -> None:
        radius = self.media_radius if not hero else self._s(30)
        self._draw_card_shell(surf, rect, fill_color=self.palette["card"], radius=radius)
        inset = self._s(3)
        image_rect = rect.inflate(-inset * 2, -inset * 2)
        image = self._load_rounded_cover(image_path, image_rect.size, radius=max(8, radius - inset))
        surf.blit(image, image_rect.topleft)

    def _draw_details_card(self, surf, rect, profile: HingeProfile) -> None:
        self._draw_card_shell(surf, rect, fill_color=self.palette["card_alt"])
        x = rect.x + self._s(22)
        table_top = rect.y + self._s(20)
        row_specs = [
            ("Name", _truncate_chars(profile.name, 30), self.table_value_emphasis_font, 1),
            ("Age", str(profile.age), self.table_value_font, 1),
            ("Occupation", _truncate_chars(profile.occupation, 34), self.table_value_font, 1),
            ("Dating intentions", profile.dating_intentions, self.table_value_font, 2),
        ]
        label_x = x
        value_x = rect.x + self._s(176)
        value_width = rect.right - self._s(22) - value_x
        row_heights = [self._s(18), self._s(18), self._s(18), self._s(28)]
        row_gap = self._s(5)

        cursor_y = table_top
        for idx, (label, value, value_font, max_lines) in enumerate(row_specs):
            label_img = self.table_label_font.render(label, True, self.palette["muted"])
            surf.blit(label_img, (label_x, cursor_y + self._s(1)))

            value_lines = self._wrap_lines(value_font, value, value_width, max_lines=max_lines)
            line_gap = self._s(2) if max_lines > 1 else 0
            self._blit_lines(
                surf,
                value_font,
                value_lines,
                self.palette["text"],
                (value_x, cursor_y),
                line_gap=line_gap,
            )

            row_bottom = cursor_y + row_heights[idx]
            if idx < len(row_specs) - 1:
                rule_y = row_bottom + self._s(4)
                self.pygame.draw.line(
                    surf,
                    self.palette["rule"],
                    (label_x, rule_y),
                    (rect.right - self._s(22), rule_y),
                    1,
                )
            cursor_y = row_bottom + row_gap

    def _build_profile_surface(self, profile: HingeProfile) -> None:
        pygame = self.pygame
        w, h = self.profile_rect.size
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        self._rounded(
            surf,
            pygame.Rect(0, 0, w, h),
            self.palette["board"],
            radius=self.board_radius,
            border=1,
            border_color=self.palette["board_line"],
        )

        accent_wash = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(
            accent_wash,
            (*self.palette["accent_soft"], 65),
            pygame.Rect(self._s(-24), self._s(-40), self._s(360), self._s(220)),
        )
        pygame.draw.ellipse(
            accent_wash,
            (255, 255, 255, 120),
            pygame.Rect(self._s(760), self._s(350), self._s(260), self._s(160)),
        )
        surf.blit(accent_wash, (0, 0))

        left_prompt = self.left_prompt_rect.move(-self.profile_rect.x, -self.profile_rect.y)
        hero_photo = self.center_photo_rect.move(-self.profile_rect.x, -self.profile_rect.y)
        right_prompt = self.right_prompt_rect.move(-self.profile_rect.x, -self.profile_rect.y)
        left_photo = self.left_photo_rect.move(-self.profile_rect.x, -self.profile_rect.y)
        meta = self.meta_rect.move(-self.profile_rect.x, -self.profile_rect.y)
        right_photo = self.right_photo_rect.move(-self.profile_rect.x, -self.profile_rect.y)

        self._draw_prompt_card(surf, left_prompt, profile.prompt_1)
        self._draw_media_card(surf, hero_photo, profile.picture_2, hero=True)
        self._draw_hero_text(surf, profile, hero_photo.inflate(-self._s(6), -self._s(6)).move(self._s(3), self._s(3)))
        self._draw_prompt_card(surf, right_prompt, profile.prompt_2)
        self._draw_media_card(surf, left_photo, profile.picture_1)
        self._draw_details_card(surf, meta, profile)
        self._draw_media_card(surf, right_photo, profile.picture_3)

        self.profile_surface_shadow = self._make_shadow_surface(
            (w, h),
            radius=self.board_radius,
            layers=[(34, 8, 2), (16, 14, 8), (6, 22, 14)],
        )
        self.profile_surface = surf

    def show_profile(self, profile: HingeProfile) -> None:
        self._build_profile_surface(profile)
        self.profile_offset_x = 0
        self.profile_opacity = 255
        self.profile_visible = True
        self.overlay_kind = None
        self.overlay_alpha = 0
        self.render()

    def set_status(self, command: str, sub: str = "", footer: str | None = None) -> None:
        self.command_text = command
        self.sub_text = sub
        if footer is not None:
            self.footer_text = footer

    def set_transform(self, dx: float = 0.0, opacity: float = 1.0) -> None:
        self.profile_offset_x = int(round(dx))
        self.profile_opacity = max(0, min(255, int(round(float(opacity) * 255.0))))
        self.profile_visible = self.profile_opacity > 6

    def set_overlay(self, overlay: str | None, alpha: float = 0.0) -> None:
        self.overlay_kind = overlay
        self.overlay_alpha = max(0, min(255, int(round(alpha * 255.0))))

    def consume_keys(self) -> list[str]:
        pygame = self.pygame
        keys: list[str] = []
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                self.closed = True
                keys.append("escape")
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_ESCAPE:
                    self.closed = True
                    keys.append("escape")
                elif evt.key == pygame.K_LEFT:
                    keys.append("left")
                elif evt.key == pygame.K_RIGHT:
                    keys.append("right")
                elif evt.key == pygame.K_SPACE:
                    keys.append("space")
        self._keys = keys
        return keys

    def render(self) -> None:
        pygame = self.pygame
        self.screen.blit(self.background_surface, (0, 0))

        brand_shadow = self._make_shadow_surface(self.brand_rect.size, radius=self._s(22), layers=[(16, 6, 2), (6, 12, 6)])
        brand_pad_x = (brand_shadow.get_width() - self.brand_rect.width) // 2
        brand_pad_y = max(0, (brand_shadow.get_height() - self.brand_rect.height) // 2 - self._s(3))
        self.screen.blit(brand_shadow, (self.brand_rect.x - brand_pad_x, self.brand_rect.y - brand_pad_y))
        self._rounded(
            self.screen,
            self.brand_rect,
            self.palette["card"],
            radius=self._s(22),
            border=1,
            border_color=self.palette["board_line"],
        )
        badge = self.logo_font.render("Honge", True, (83, 57, 50))
        self.screen.blit(
            badge,
            (self.brand_rect.centerx - badge.get_width() // 2, self.brand_rect.centery - badge.get_height() // 2 - self._s(1)),
        )
        self.pygame.draw.circle(
            self.screen,
            self.palette["accent"],
            (self.brand_rect.right - self._s(22), self.brand_rect.centery),
            self._s(4),
        )

        if self.command_text.strip() or self.sub_text.strip():
            status_shadow = self._make_shadow_surface(
                self.status_rect.size,
                radius=self._s(24),
                layers=[(18, 8, 2), (8, 16, 8)],
            )
            status_pad_x = (status_shadow.get_width() - self.status_rect.width) // 2
            status_pad_y = max(0, (status_shadow.get_height() - self.status_rect.height) // 2 - self._s(4))
            self.screen.blit(status_shadow, (self.status_rect.x - status_pad_x, self.status_rect.y - status_pad_y))
            self._rounded(
                self.screen,
                self.status_rect,
                self.palette["card"],
                radius=self._s(24),
                border=1,
                border_color=self.palette["card_line"],
            )
            if self.command_text.strip():
                cmd_lines = self._wrap_lines(
                    self.status_title_font,
                    self.command_text,
                    self.status_rect.width - self._s(60),
                    max_lines=2,
                )
                cmd_height = self._s(24 if len(cmd_lines) == 1 else 38)
                cmd_top = self.status_rect.y + (self._s(7) if self.sub_text.strip() else (self.status_rect.height - cmd_height) // 2)
                self._draw_centered_lines(
                    self.screen,
                    self.status_title_font,
                    cmd_lines,
                    self.palette["text"],
                    self.pygame.Rect(
                        self.status_rect.x,
                        cmd_top,
                        self.status_rect.width,
                        cmd_height,
                    ),
                    line_gap=self._s(1),
                )
                sub_top = cmd_top + cmd_height - self._s(2)
            else:
                sub_top = self.status_rect.y + self._s(18)
            if self.sub_text.strip():
                sub_lines = self._wrap_lines(
                    self.status_body_font,
                    self.sub_text,
                    self.status_rect.width - self._s(60),
                    max_lines=2,
                )
                self._draw_centered_lines(
                    self.screen,
                    self.status_body_font,
                    sub_lines,
                    self.palette["muted"],
                    self.pygame.Rect(
                        self.status_rect.x,
                        sub_top,
                        self.status_rect.width,
                        self.status_rect.bottom - sub_top,
                    ),
                    line_gap=0,
                )

        if self.profile_surface is not None and self.profile_visible:
            offset_x = self.profile_rect.x + self.profile_offset_x
            angle = max(-7.0, min(7.0, self.profile_offset_x / max(1.0, float(self._s(32)))))
            scale = 1.0 - min(0.025, abs(self.profile_offset_x) / max(1.0, float(self._s(32000))))

            card = self.profile_surface.copy()
            card.set_alpha(self.profile_opacity)
            shadow = self.profile_surface_shadow.copy()
            shadow.set_alpha(max(0, min(255, int(round(self.profile_opacity * 0.70)))))

            if abs(angle) > 0.05 or abs(scale - 1.0) > 0.001:
                card = pygame.transform.rotozoom(card, -angle, scale)
                shadow = pygame.transform.rotozoom(shadow, -angle, scale)

            shadow_pos = (
                offset_x + self.profile_rect.width // 2 - shadow.get_width() // 2,
                self.profile_rect.y + self.profile_rect.height // 2 - shadow.get_height() // 2 + self._s(6),
            )
            card_pos = (
                offset_x + self.profile_rect.width // 2 - card.get_width() // 2,
                self.profile_rect.y + self.profile_rect.height // 2 - card.get_height() // 2,
            )
            self.screen.blit(shadow, shadow_pos)
            self.screen.blit(card, card_pos)

        if self.overlay_kind is not None and self.overlay_alpha > 0:
            overlay_label = "MATCH" if self.overlay_kind == "right" else "PASS"
            overlay_color = self.palette["accent"] if self.overlay_kind == "right" else (62, 53, 51)
            stamp = pygame.Surface((self._s(252), self._s(96)), pygame.SRCALPHA)
            self._rounded(
                stamp,
                stamp.get_rect(),
                (255, 255, 255, 228),
                radius=self._s(26),
                border=self._s(3),
                border_color=overlay_color,
            )
            label = self.overlay_font.render(overlay_label, True, overlay_color)
            stamp.blit(
                label,
                (stamp.get_width() // 2 - label.get_width() // 2, stamp.get_height() // 2 - label.get_height() // 2),
            )
            stamp = pygame.transform.rotozoom(stamp, -12 if self.overlay_kind == "right" else 11, 1.0)
            stamp.set_alpha(self.overlay_alpha)
            self.screen.blit(
                stamp,
                (self.win_size[0] // 2 - stamp.get_width() // 2, self.win_size[1] // 2 - stamp.get_height() // 2 - self._s(18)),
            )

        if self.footer_text.strip():
            footer_img = self.footer_font.render(self.footer_text, True, self.palette["muted"])
            self.screen.blit(footer_img, (self.footer_rect.centerx - footer_img.get_width() // 2, self.footer_rect.y))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self) -> None:
        self.pygame.display.quit()
        self.pygame.quit()


def _create_calibration_window(task_cfg: HingeTaskConfig) -> tuple[visual.Window, dict[str, object]]:
    local_win = visual.Window(
        size=task_cfg.win_size,
        color=(0.985, 0.962, 0.950),
        units="pix",
        fullscr=task_cfg.fullscreen,
    )
    chrome = {
        "command_shadow": visual.Rect(local_win, width=1110, height=70, pos=(0.0, 320.0), fillColor=(0.91, 0.87, 0.85), lineColor=None, opacity=0.24),
        "command_box": visual.Rect(local_win, width=1096, height=64, pos=(0.0, 324.0), fillColor=(0.995, 0.992, 0.989), lineColor=(0.90, 0.86, 0.84), lineWidth=1.8),
        "command_text": visual.TextStim(local_win, text="", pos=(0.0, 336.0), height=22, color=(-0.34, -0.22, -0.08), bold=True, wrapWidth=980),
        "sub_text": visual.TextStim(local_win, text="", pos=(0.0, 312.0), height=13, color=(-0.55, -0.47, -0.36), wrapWidth=990),
        "footer": visual.TextStim(local_win, text="Press SPACE to continue. ESC quits.", pos=(0.0, -366.0), height=16, color=(-0.44, -0.34, -0.22), wrapWidth=1080),
    }
    return local_win, chrome


def preview_profiles(profile_name: str | None = None) -> None:
    task_cfg = HingeTaskConfig()
    cfgs = apply_runtime_config_overrides("hinge_task", task_cfg=task_cfg)
    task_cfg = cfgs["task_cfg"]

    profiles = _load_profiles(task_cfg.profiles_dir)
    if profile_name is not None:
        key = str(profile_name).strip().lower()
        profiles = [p for p in profiles if p.folder_name.lower() == key]
        if not profiles:
            raise RuntimeError(f"No profile folder named {profile_name!r} found under {task_cfg.profiles_dir}")

    ui = HongePygameUI(task_cfg)
    profile_index = 0
    try:
        while True:
            ui.show_profile(profiles[profile_index])
            while True:
                ui.set_status("", "", "Use LEFT/RIGHT to browse profiles." if len(profiles) > 1 else "")
                ui.render()
                keys = ui.consume_keys()
                if "escape" in keys:
                    return
                if "right" in keys and len(profiles) > 1:
                    profile_index = (profile_index + 1) % len(profiles)
                    break
                if "left" in keys and len(profiles) > 1:
                    profile_index = (profile_index - 1) % len(profiles)
                    break
    finally:
        ui.close()


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
    if missing:
        raise RuntimeError(
            f"Live stream is missing configured EEG picks {missing}. "
            f"Configured picks: {list(eeg_cfg.picks)}. Available channels: {available}"
        )
    if len(model_ch_names) < 2:
        raise RuntimeError(
            "Need at least 2 configured EEG channels after applying picks. "
            f"Configured picks: {list(eeg_cfg.picks)}. Resolved channels: {model_ch_names}. "
            f"Available channels: {available}"
        )
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
    ui: HongePygameUI | None = None
    calib_win: visual.Window | None = None

    def _draw(*, command: str, sub: str = "", dx: float = 0.0, alpha: float = 1.0, overlay: str | None = None, overlay_alpha: float = 0.0) -> None:
        if ui is None:
            return
        ui.set_status(command, sub, "Jaw clench to start a swipe trial. ESC quits.")
        ui.set_transform(dx=dx, opacity=alpha)
        ui.set_overlay(overlay, alpha=overlay_alpha)
        ui.render()

    def _show_completion_screen() -> None:
        if ui is None:
            return
        left_swipes = sum(1 for result in trial_results if str(result.get("swipe_direction")) == "left")
        right_swipes = sum(1 for result in trial_results if str(result.get("swipe_direction")) == "right")
        ui.profile_visible = False
        ui.set_overlay(None, alpha=0.0)
        ui.set_status(
            "Session complete",
            f"Swiped left: {left_swipes}   Swiped right: {right_swipes}",
            "Press ESC to exit task.",
        )
        while True:
            ui.render()
            if "escape" in ui.consume_keys():
                return

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

        jaw_idxs = select_jaw_channel_indices(model_ch_names)
        jaw_window_n = int(round(float(task_cfg.jaw_window_s) * float(sfreq)))
        jaw_classifier = None
        runtime_jaw_classifier, runtime_train_acc = resolve_runtime_jaw_classifier(
            logger=logger,
            min_total_samples=12,
            requested_channel_names=model_ch_names,
        )
        if runtime_jaw_classifier is not None:
            jaw_classifier = runtime_jaw_classifier
            logger.info("Using orchestrator-provided jaw calibration (train_acc=%.3f).", float(runtime_train_acc or 0.0))
        else:
            calib_win, chrome = _create_calibration_window(task_cfg)

            def _wait_for_space(prompt_text: str) -> None:
                while True:
                    chrome["command_text"].text = prompt_text
                    chrome["sub_text"].text = "Press SPACE to continue. ESC quits."
                    chrome["command_shadow"].draw()
                    chrome["command_box"].draw()
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    calib_win.flip()
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
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    calib_win.flip()

            def _collect_stream_block(duration_s: float) -> np.ndarray:
                def _check_abort() -> None:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt

                def _render(_elapsed_s: float, _total_s: float) -> None:
                    chrome["command_shadow"].draw()
                    chrome["command_box"].draw()
                    chrome["command_text"].draw()
                    chrome["sub_text"].draw()
                    chrome["footer"].draw()
                    calib_win.flip()

                return collect_cue_locked_stream_block(
                    stream=stream,
                    sfreq=float(sfreq),
                    n_channels=len(model_ch_names),
                    duration_s=float(duration_s),
                    cue_offset_s=float(task_cfg.special_command_cue_offset_s),
                    render_frame=_render,
                    check_abort=_check_abort,
                    logger=logger,
                    label="hinge jaw calibration block",
                )

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
                cue_offset_s=float(task_cfg.special_command_cue_offset_s),
            )
            logger.info("Runtime jaw calibration complete: train_acc=%.3f", float(jaw_train_acc))
            _close_window(calib_win)
            calib_win = None

        ui = HongePygameUI(task_cfg)
        ui.set_status("", "Jaw clench when you are ready to swipe.", "Jaw clench to start a swipe trial. ESC quits.")
        ui.render()

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
            if ui is not None and "escape" in ui.consume_keys():
                raise KeyboardInterrupt

            if ui is not None:
                ui.show_profile(profile)
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

            idle_text = "Jaw clench when you are ready to swipe."
            while True:
                if ui is not None and "escape" in ui.consume_keys():
                    raise KeyboardInterrupt
                _poll_live()
                _draw(command="", sub=f"{idle_text}  Trial {trial_idx}/{len(profile_cycle)}   jaw_p={jaw_prob:.2f}")
                if jaw_event_pending:
                    jaw_event_pending = False
                    break

            prep_clock = core.Clock()
            while prep_clock.getTime() < float(task_cfg.prep_duration_s):
                if ui is not None and "escape" in ui.consume_keys():
                    raise KeyboardInterrupt
                _poll_live()
                remaining = max(0.0, float(task_cfg.prep_duration_s) - prep_clock.getTime())
                _draw(command="Prepare to swipe left to pass or right to like", sub=f"Prepare your chosen imagery. {remaining:.1f}s remaining")

            _poll_live()
            pre_context = raw_history[:, -context_n:].copy() if context_n > 0 and raw_history.shape[1] > 0 else np.empty((len(model_ch_names), 0), dtype=np.float32)
            execute_duration_s = float(task_cfg.execute_duration_s)

            def _check_execute_abort() -> None:
                if ui is not None and "escape" in ui.consume_keys():
                    raise KeyboardInterrupt

            def _render_execute(elapsed_s: float, total_s: float) -> None:
                remaining = max(0.0, total_s - elapsed_s)
                _draw(command="Execute swipe", sub=f"Perform your chosen imagery now. {remaining:.1f}s remaining")

            # Collect the online MI epoch from a cue-locked window so the decoded
            # block consistently spans the full post-cue execute interval.
            execute_block = collect_cue_locked_stream_block(
                stream=stream,
                sfreq=float(sfreq),
                n_channels=len(model_ch_names),
                duration_s=execute_duration_s,
                cue_offset_s=0.0,
                render_frame=_render_execute,
                check_abort=_check_execute_abort,
                logger=logger,
                label=f"hinge execute block trial {trial_idx}",
            )
            raw_history = np.concatenate((raw_history, execute_block), axis=1)
            if raw_history.shape[1] > keep_n:
                raw_history = raw_history[:, -keep_n:]

            _draw(command="Stop", sub="Decoding your swipe...")
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
                if ui is not None and "escape" in ui.consume_keys():
                    raise KeyboardInterrupt
                progress = min(1.0, swipe_clock.getTime() / swipe_duration)
                eased = 1.0 - (1.0 - progress) ** 3
                dx = (780.0 * eased) * (1.0 if swipe_direction == "right" else -1.0)
                alpha = max(0.0, 1.0 - progress * 0.80)
                _draw(command="Stop", sub=f"Swipe decoded: {'LIKE' if swipe_direction == 'right' else 'PASS'}", dx=dx, alpha=alpha)

            overlay_clock = core.Clock()
            while overlay_clock.getTime() < float(task_cfg.outcome_duration_s):
                if ui is not None and "escape" in ui.consume_keys():
                    raise KeyboardInterrupt
                progress = min(1.0, overlay_clock.getTime() / float(task_cfg.outcome_duration_s))
                overlay_alpha = 1.0 - 0.45 * progress
                _draw(
                    command="Match!" if swipe_direction == "right" else "Pass",
                    sub="Swipe decoded: RIGHT" if swipe_direction == "right" else "Swipe decoded: LEFT",
                    overlay=swipe_direction,
                    overlay_alpha=overlay_alpha,
                )

            submitted_message: str | None = None
            if swipe_direction == "right" and bool(task_cfg.enable_keyboard_on_match):
                logger.info("Launching nested keyboard after right swipe for profile=%s", profile.folder_name)
                if ui is not None:
                    ui.close()
                    ui = None
                _close_stream(stream)
                stream = None
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
                ui = HongePygameUI(task_cfg)
                ui.show_profile(profile)

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
                if ui is not None and "escape" in ui.consume_keys():
                    raise KeyboardInterrupt
                if ui is not None:
                    ui.render()

        _show_completion_screen()
        with open(f"{fname}_hinge_trials.pkl", "wb") as fh:
            pickle.dump(trial_results, fh)
        logger.info("Saved %d hinge trials to %s_hinge_trials.pkl", len(trial_results), fname)

    except KeyboardInterrupt:
        logger.info("Hinge task interrupted by user.")
    finally:
        _close_stream(stream)
        _close_window(calib_win)
        if ui is not None:
            ui.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honge BCI task")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the Honge profile GUI without EEG, models, or jaw clench flow.",
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
