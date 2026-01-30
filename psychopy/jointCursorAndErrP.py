# Open this with PsychoPy
from psychopy import visual, core, event
import random
import serial

# --- Configuration ---
NUM_TRIALS = 50
WIN_SIZE = [1000, 600]
TARGET_OFFSET = 0.6  # Normalized units (0.6 is 60% of the way to the edge)
ACCURACY_RATE = 0.75

# --- Timing Configuration (Seconds) ---
IMAGERY_DURATION = 2.0  # How long the target flashes
MOVE_DURATION = 1.0     # How long the cursor takes to move
ITI_DURATION = 2.0      # Inter-trial interval

# Trigger Hub Config
PORT = 'COM5'
LEFT_TRIGGER = 1
RIGHT_TRIGGER = 2
NO_ERROR_TRIGGER = 3
ERRP_TRIGGER = 4

mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()

# --- Setup Window ---
win = visual.Window(size=WIN_SIZE, color='black', units='norm', fullscr=False)

# --- Define Stimuli ---
cursor = visual.Circle(win, radius=0.05, fillColor='white', lineColor='white', pos=(0, 0))

target_left = visual.Circle(win, radius=0.10, fillColor=None, lineColor='white',
                            lineWidth=3, pos=(-TARGET_OFFSET, 0))
target_right = visual.Circle(win, radius=0.10, fillColor=None, lineColor='white',
                             lineWidth=3, pos=(TARGET_OFFSET, 0))

instr_text = visual.TextStim(win, text="", pos=(0, 0.6), height=0.12, wrapWidth=1.8)

# --- Main Trial Loop ---
for trial in range(NUM_TRIALS):

    if 'escape' in event.getKeys():
        break

    # 1. SETUP LOGIC
    target_side = random.choice(['left', 'right'])
    side_trigger = LEFT_TRIGGER if target_side == 'left' else RIGHT_TRIGGER

    if random.random() < ACCURACY_RATE:
        move_side = target_side
        outcome_trigger = NO_ERROR_TRIGGER
    else:
        move_side = 'left' if target_side == 'right' else 'right'
        outcome_trigger = ERRP_TRIGGER

    # 2. PART A: IMAGERY INSTRUCTION (Text Only)
    instr_text.text = "Target about to flash!\nWhen it does, imagine moving the cursor to it."

    target_left.fillColor = None
    target_right.fillColor = None
    cursor.pos = (0, 0)

    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()
    win.flip()

    core.wait(random.uniform(1.5, 2.5))

    # 3. PART B: IMAGERY ACTION (Flash) + LEFT/RIGHT trigger aligned to flash flip
    if target_side == 'left':
        target_left.fillColor = 'green'
        target_right.fillColor = None
    else:
        target_right.fillColor = 'green'
        target_left.fillColor = None

    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()

    # Send cue trigger exactly on the flip that shows the flash
    win.callOnFlip(mmbts.write, bytes([side_trigger]))
    win.flip()

    core.wait(IMAGERY_DURATION)

    # 4. MOVEMENT ACTION (Animation) + outcome trigger aligned to FIRST movement flip
    start_pos = 0.0
    end_pos = -TARGET_OFFSET if move_side == 'left' else TARGET_OFFSET

    # ---- First movement frame (separate) ----
    # Start a clock *before* scheduling the first movement flip
    move_clock = core.Clock()

    # Compute a tiny positive t for the first movement frame so cursor actually moves on that first flip.
    # Using the monitor frame period is a good approximation; if unavailable, fall back to ~1/60.
    frame_period = win.monitorFramePeriod if win.monitorFramePeriod else (1.0 / 60.0)
    t0 = frame_period

    current_x = start_pos + (end_pos - start_pos) * (t0 / MOVE_DURATION)
    cursor.pos = (current_x, 0)

    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()

    # Send outcome trigger exactly on the flip that initiates visible movement
    win.callOnFlip(mmbts.write, bytes([outcome_trigger]))
    win.flip()

    # ---- Remaining movement frames ----
    while True:
        t = move_clock.getTime()
        if t >= MOVE_DURATION:
            break

        current_x = start_pos + (end_pos - start_pos) * (t / MOVE_DURATION)
        cursor.pos = (current_x, 0)

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()
        win.flip()

        if 'escape' in event.getKeys():
            win.close()
            core.quit()

    # Turn off flash
    target_left.fillColor = None
    target_right.fillColor = None

    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()
    win.flip()

    core.wait(ITI_DURATION)

# --- Cleanup ---
win.close()
core.quit()
