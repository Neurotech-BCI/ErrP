# Open this with PsychoPy
from psychopy import visual, core, event
import random
import serial 
# --- Configuration ---
NUM_TRIALS = 5
WIN_SIZE = [1000, 600]
TARGET_OFFSET = 0.6  # Normalized units (0.6 is 60% of the way to the edge)
ACCURACY_RATE = 0.5 

# --- Timing Configuration (Seconds) ---
TEXT_READ_DELAY = 2.0   # Delay between first text appearing and action starting
IMAGERY_DURATION = 2.0  # How long the target flashes
MOVE_DURATION = 1.0     # How long the cursor takes to move
ITI_DURATION = 0.5      # Inter-trial interval

# Trigger Hub Config
PORT = 'COM5'
CURSOR_TRIGGER = 1
ERRP_TRIGGER = 2

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
    
    # Check for quit early
    if 'escape' in event.getKeys():
        break

    # 1. SETUP LOGIC
    # --------------
    target_side = random.choice(['left', 'right'])
    
    # Determine movement logic
    if random.random() < ACCURACY_RATE:
        move_side = target_side
    else:
        move_side = 'left' if target_side == 'right' else 'right'

    # 2. PART A: IMAGERY INSTRUCTION (Text Only)
    # ------------------------------------------
    instr_text.text = "Target about to flash!\nWhen it does, imagine moving the cursor to it."
    
    # Ensure targets are neutral (no flash yet)
    target_left.fillColor = None
    target_right.fillColor = None
    cursor.pos = (0, 0)
    
    # Draw scene
    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()
    win.flip()
    
    # Wait for user to read
    core.wait(TEXT_READ_DELAY)

    # 3. PART B: IMAGERY ACTION (Flash)
    # ---------------------------------
    # Set flash color
    if target_side == 'left':
        target_left.fillColor = 'green'
    else:
        target_right.fillColor = 'green'
        
    # Draw scene with flash
    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()
    win.flip()
    
    # Write event trigger for target appearing
    mmbts.write(bytes([CURSOR_TRIGGER]))
    
    # Hold the flash
    core.wait(IMAGERY_DURATION)


    # 5. PART D: MOVEMENT ACTION (Animation)
    # --------------------------------------
    start_pos = 0 
    end_pos = -TARGET_OFFSET if move_side == 'left' else TARGET_OFFSET
    
    move_clock = core.Clock()
    
    # Write Errp event trigger
    mmbts.write(bytes([ERRP_TRIGGER]))
    
    while move_clock.getTime() < MOVE_DURATION:
        t = move_clock.getTime()
        
        # Calculate position: start + (distance * (time / duration))
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
    
    # Draw scene
    target_left.draw()
    target_right.draw()
    cursor.draw()
    instr_text.draw()
    win.flip()
    # 6. INTER-TRIAL INTERVAL
    # -----------------------
    core.wait(ITI_DURATION)

# --- Cleanup ---
win.close()
core.quit()