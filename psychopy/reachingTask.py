import os
import logging
import platform
import random
import serial
from psychopy import visual, core, event

# --- Block terminal noise ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('matplotlib.font_manager').disabled = True

# --- 1. Configuration ---
if platform.system() == "Windows":
    PORT = "COM6"
else:
    PORT = "/dev/cu.usbserial-10" 

NUM_TRIALS = 50
RUNS_PER_LOOP = 3
BREAK_DURATION = 60

# Timing aligned to your joint task
MOVE_DURATION = 2.5 # Total imagery movement time                     
ITI_DURATION = 1.5                      

TRIG_LEFT_ARM = 10
TRIG_RIGHT_ARM = 20
TRIG_REST = 30

# --- 2. Initialize Serial (Safe Mode) ---
try:
    mmbts = serial.Serial(PORT, baudrate=115200, timeout=0.1)
    print(f"SUCCESS: Connected on {PORT}")
except:
    mmbts = None
    print("OFFLINE MODE: No EEG box found. Running visuals only.")

# --- 3. Setup Window ---
win = visual.Window(color="gray", units='norm', fullscr=True)

target_left = visual.Circle(win, radius=0.15, lineColor='black', lineWidth=3, pos=(-0.6, 0))
target_right = visual.Circle(win, radius=0.15, lineColor='black', lineWidth=3, pos=(0.6, 0))

instr_text = visual.TextStim(win, text="", pos=(0, 0.6), color="black", height=0.08, wrapWidth=1.8)
break_text = visual.TextStim(win, text="", color="black", height=0.1)

# --- 4. Main Loop ---
for run in range(RUNS_PER_LOOP):
    for trial in range(NUM_TRIALS):
        if 'escape' in event.getKeys(): core.quit()

        limb = random.choice(['LEFT', 'RIGHT', 'REST'])
        direction = random.choice(['left', 'right']) 
        
        if limb == 'LEFT': side_trigger = TRIG_LEFT_ARM
        elif limb == 'RIGHT': side_trigger = TRIG_RIGHT_ARM
        else: side_trigger = TRIG_REST

        target_obj = target_left if direction == 'left' else target_right
        target_pos = -0.6 if direction == 'left' else 0.6

        cursor.pos = (0, 0)
        target_left.fillColor = target_right.fillColor = None
        
        if limb == 'REST':
            instr_text.text = "Resting: Keep your arms still."
            current_trigger = TRIG_REST
        else:
            instr_text.text = f"Prepare to REACH with your {limb} ARM toward the {direction.upper()} target."
        
        target_left.draw(); target_right.draw(); instr_text.draw()
        win.flip()
        core.wait(random.uniform(1.5, 2.5))

        if limb != 'REST':
            target_obj.fillColor = 'green'
            instr_text.text = f"IMAGINE REACHING: {limb} ARM"
        
        if mmbts:
            win.callOnFlip(mmbts.write, bytes([current_trigger]))
        
        target_left.draw(); target_right.draw(); instr_text.draw()
        win.flip()
        core.wait(0.5)

        # PHASE 3: MOVEMENT DURATION
        move_timer = core.Clock()
        while move_timer.getTime() < MOVE_DURATION:
            if limb != 'REST':
                progress = move_timer.getTime() / MOVE_DURATION
                cursor.pos = (target_pos * progress, 0)
            
            target_left.draw(); target_right.draw(); cursor.draw(); instr_text.draw()
            win.flip()

        win.flip()
        core.wait(ITI_DURATION)

    # BREAK
    if run != RUNS_PER_LOOP - 1:
        timer = core.CountdownTimer(BREAK_DURATION)
        while timer.getTime() > 0:
            if 'escape' in event.getKeys(): core.quit()
            break_text.text = f"Break time!\n{int(timer.getTime())}s"
            break_text.draw()
            win.flip()

win.close()
core.quit()