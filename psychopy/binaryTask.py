# This is just the joint task without the ErrP trials
from psychopy import visual, core, event
import random
import serial

# --- Configuration ---
NUM_TRIALS = 50
WIN_SIZE = [1000, 600]  # not used anymore because of fullscreen
TARGET_OFFSET = 0.6  # Normalized units (0.6 is 60% of the way to the edge)

# --- Timing Configuration (Seconds) ---
IMAGERY_DURATION = 2.0  # How long the target flashes
MOVE_DURATION = 1.0     # How long the cursor takes to move
ITI_DURATION = 1.5      # Inter-trial interval

RUNS_PER_LOOP = 3 # each run is 50 trials (or whatever NUM_TRIALS is set to)
BREAK_DURATION = 60 # 60 second break between runs

# Trigger Hub Config - use COM3 on the Ahmanson desktop
PORT = 'COM6'
LEFT_TRIGGER = 1
RIGHT_TRIGGER = 2
REST_TRIGGER = 3

mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()

# --- Setup Window ---
win = visual.Window(color='black', units='norm', fullscr=True)

# --- Define Stimuli ---
cursor = visual.Circle(win, radius=0.05, fillColor='white', lineColor='white', pos=(0, 0))

target_left = visual.Circle(win, radius=0.10, fillColor=None, lineColor='white',
                            lineWidth=3, pos=(-TARGET_OFFSET, 0))
target_right = visual.Circle(win, radius=0.10, fillColor=None, lineColor='white',
                             lineWidth=3, pos=(TARGET_OFFSET, 0))

instr_text = visual.TextStim(win, text="", pos=(0, 0.6), height=0.12, wrapWidth=1.8)
break_text = visual.TextStim(win, text="", pos=(0, 0), height=0.12, wrapWidth=1.8)

# -- Session Loop --
for run in range(RUNS_PER_LOOP):
    # --- Run Loop ---
    for trial in range(NUM_TRIALS):

        if 'escape' in event.getKeys():
            break

        # 1. SETUP LOGIC
        targets_list = ['left', 'right', 'rest']
        target_side = random.choice(targets_list)
        if target_side == 'left':
          side_trigger = LEFT_TRIGGER
        elif target_side == 'right':
          side_trigger = RIGHT_TRIGGER
        else:
          side_trigger = REST_TRIGGER

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
        elif target_side == 'right':
            target_right.fillColor = 'green'
            target_left.fillColor = None
        else:
            target_right.fillColor = None
            target_left.fillColor = None

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()

        # Send cue trigger exactly on the flip that shows the flash
        win.callOnFlip(mmbts.write, bytes([side_trigger]))
        win.flip()

        core.wait(IMAGERY_DURATION)
        win.flip()

        # Turn off flash
        target_left.fillColor = None
        target_right.fillColor = None

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()
        win.flip()

        core.wait(ITI_DURATION)
    
    win.flip()
    if run != RUNS_PER_LOOP - 1:
        timer = core.CountdownTimer(BREAK_DURATION)
        while timer.getTime() > 0:  # after 5s will become negative
            break_text.text = f"Break time!\nPlease come back in {int(timer.getTime())}s"
            break_text.draw()
            win.flip()
# --- Cleanup ---
win.close()
core.quit()
mmbts.close()
