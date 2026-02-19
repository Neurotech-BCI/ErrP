from psychopy import visual, core, event
import numpy as np
import time
import serial

# --- Configuration ---
WIN_SIZE = [1200, 800]
CURSOR_SIZE = 15
TARGET_SIZE = 40

# Movement Settings
STEP_SIZE = 60                   # Distance of one "move"
TARGET_DISTANCE = 5 * STEP_SIZE  # Target is 5 steps away
MOVE_DURATION = 1.25             # Seconds it takes to glide to next point (Continuous movement)

# Error Parameters
ERROR_PROB = 0.25

# Experiment Settings
N_TRIALS = 50

# --- Trigger Port (commented out for testing) ---
PORT = 'COM6'

# 4 triggers total:
NON_ERROR_TRIGGER = 1                 # baseline / no error
TOWARD_TARGET_ERROR_TRIGGER = 2       # error movement still reduced distance to target
PERPENDICULAR_ERROR_TRIGGER = 3       # perpendicular error that does NOT reduce distance
OPPOSITE_ERROR_TRIGGER = 4            # opposite error (away from intended direction)

mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()

# --- Setup ---
win = visual.Window(size=WIN_SIZE, units='pix', color='black')

cursor = visual.Circle(win, radius=CURSOR_SIZE, fillColor='blue', lineColor='white')
target = visual.Rect(win, width=TARGET_SIZE, height=TARGET_SIZE, fillColor='red', lineColor='white')

instr = visual.TextStim(win, text="", color='white', height=30, pos=(0, 350))

# --- Helper Functions ---
def get_movement_vector(keys, step):
    dx, dy = 0, 0
    if 'up' in keys:
        dy = step
    elif 'down' in keys:
        dy = -step
    elif 'left' in keys:
        dx = -step
    elif 'right' in keys:
        dx = step
    return np.array([dx, dy], dtype=float)

def perpendicular_vectors(vec):
    """
    For an axis-aligned movement vector [dx, dy], return the two perpendicular options:
    +90: [-dy, dx], -90: [dy, -dx]
    """
    dx, dy = vec[0], vec[1]
    return (np.array([-dy, dx], dtype=float), np.array([dy, -dx], dtype=float))

def distance(a, b):
    return float(np.linalg.norm(a - b))

# --- Experiment Loop ---
start_msg = visual.TextStim(
    win,
    text="Reach the Red Square.\n\nPress arrow keys to move.\n\nPress any key to start.",
    color='white'
)
start_msg.draw()
win.flip()
event.waitKeys()

for trial_i in range(N_TRIALS):

    # 1. Reset Trial
    cursor.pos = (0, 0)

    # Position Target
    target_angle = np.random.uniform(0, 2 * np.pi)
    target.pos = (TARGET_DISTANCE * np.cos(target_angle), TARGET_DISTANCE * np.sin(target_angle))

    trial_complete = False

    # 2. Movement Loop
    while not trial_complete:

        # Draw static elements
        target.draw()
        cursor.draw()
        instr.text = f"Trial {trial_i + 1}/{N_TRIALS}"
        instr.draw()
        win.flip()

        # Check Inputs
        keys = event.waitKeys(keyList=['up', 'down', 'left', 'right', 'escape'])

        if 'escape' in keys:
            win.close()
            core.quit()

        if not keys:
            continue

        # --- Calculate Intended Movement ---
        start_pos = np.array(cursor.pos, dtype=float)
        intended_vec = get_movement_vector(keys, STEP_SIZE)

        if np.allclose(intended_vec, 0):
            continue

        # Distance to target before movement (used for "toward-target error" classification)
        target_pos = np.array(target.pos, dtype=float)
        dist_before = distance(start_pos, target_pos)

        # --- Error Logic ---
        is_error = (np.random.random() < ERROR_PROB)

        move_vec = intended_vec.copy()
        trigger_to_send = NON_ERROR_TRIGGER
        if is_error:
            # 50% opposite, 50% perpendicular. If perpendicular, choose either perpendicular direction with 50/50.
            if np.random.random() < 0.5:
                # Opposite
                error_type = "opposite"
                move_vec = -intended_vec
            else:
                # Perpendicular
                error_type = "perpendicular"
                perp_a, perp_b = perpendicular_vectors(intended_vec)
                move_vec = perp_a if (np.random.random() < 0.5) else perp_b

            # Classify whether this *error movement* still brings cursor closer to target
            end_pos_candidate = start_pos + move_vec
            dist_after = distance(end_pos_candidate, target_pos)

            if dist_after < dist_before:
                trigger_to_send = TOWARD_TARGET_ERROR_TRIGGER
            else:
                if error_type == "perpendicular":
                    trigger_to_send = PERPENDICULAR_ERROR_TRIGGER
                else:
                    trigger_to_send = OPPOSITE_ERROR_TRIGGER
        # --- Final Destination (with boundary clamp) ---
        end_pos = start_pos + move_vec
        end_pos[0] = np.clip(end_pos[0], -WIN_SIZE[0] // 2, WIN_SIZE[0] // 2)
        end_pos[1] = np.clip(end_pos[1], -WIN_SIZE[1] // 2, WIN_SIZE[1] // 2)

        # Queue trigger to be sent on the very next screen flip (i.e., movement onset)
        win.callOnFlip(mmbts.write, bytes([trigger_to_send]))

        # --- Animate Movement (Visual) ---
        animation_clock = core.Clock()
        while animation_clock.getTime() < MOVE_DURATION:
            t = animation_clock.getTime() / MOVE_DURATION

            current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
            current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * t
            cursor.pos = (current_x, current_y)

            target.draw()
            cursor.draw()
            instr.draw()
            win.flip()

        # Ensure final position is exact
        cursor.pos = end_pos

        # --- Check Success (Proper Circle-Rectangle Collision) ---

        cursor_center = np.array(cursor.pos, dtype=float)
        target_center = np.array(target.pos, dtype=float)

        half_w = TARGET_SIZE / 2
        half_h = TARGET_SIZE / 2

        # Find closest point on rectangle to circle center
        closest_x = np.clip(cursor_center[0],
                            target_center[0] - half_w,
                            target_center[0] + half_w)

        closest_y = np.clip(cursor_center[1],
                            target_center[1] - half_h,
                            target_center[1] + half_h)

        closest_point = np.array([closest_x, closest_y])

        dist_to_rect = np.linalg.norm(cursor_center - closest_point)

        if dist_to_rect < CURSOR_SIZE:
            trial_complete = True

            feedback = visual.TextStim(win, text="Target Reached!", color='green', height=40)
            for _ in range(60):  # ~1 sec at 60 Hz
                target.draw()
                cursor.draw()
                feedback.draw()
                win.flip()

# --- End ---
win.close()