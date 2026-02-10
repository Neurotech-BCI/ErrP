from psychopy import visual, core, event
import random
import serial
import platform

# --- 1. Multi-Platform & Offline Configuration ---
if platform.system() == "Windows":
    PORT = "COM6"
else:
    # MacBook port: check with 'ls /dev/cu.usb*'
    PORT = "/dev/cu.usbserial-10" 

NUM_TRIALS = 50
RUNS_PER_LOOP = 3
BREAK_DURATION = 60

PREP_DURATION = random.uniform(1.5, 2.5) 
FLASH_ONLY_DURATION = 0.5               
MOVE_DURATION = 2.5                     
ITI_DURATION = 1.5                      

TRIG_LEFT_ARM = 10
TRIG_RIGHT_ARM = 20
TRIG_REST = 30

# This block allows the script to run WITHOUT an EEG box
try:
    mmbts = serial.Serial(PORT, baudrate=115200, timeout=1)
    print(f"SUCCESS: Connected to EEG box on {PORT}")
except:
    mmbts = None
    print("OFFLINE MODE: No EEG box found. Triggers will not be sent, but task will run.")

win = visual.Window(color="gray", units='norm', fullscr=True)

cursor = visual.Circle(win, radius=0.05, fillColor='black', pos=(0, 0))
target_left = visual.Circle(win, radius=0.15, lineColor='black', lineWidth=3, pos=(-0.6, 0))
target_right = visual.Circle(win, radius=0.15, lineColor='black', lineWidth=3, pos=(0.6, 0))

instr_text = visual.TextStim(win, text="", pos=(0, 0.6), color="black", height=0.08, wrapWidth=1.8)
break_text = visual.TextStim(win, text="", color="black", height=0.1)

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
        else:
            instr_text.text = f"Prepare to REACH with your {limb} ARM toward the {direction.upper()} target."
        
        target_left.draw(); target_right.draw(); cursor.draw(); instr_text.draw()
        win.flip()
        core.wait(PREP_DURATION)

        if limb != 'REST':
            target_obj.fillColor = 'green'
            instr_text.text = f"IMAGINE REACHING: {limb} ARM"
        
        # This only executes if mmbts was successfully connected
        if mmbts:
            win.callOnFlip(mmbts.write, bytes([side_trigger]))
        
        target_left.draw(); target_right.draw(); cursor.draw(); instr_text.draw()
        win.flip()
        core.wait(FLASH_ONLY_DURATION)

        move_timer = core.Clock()
        while move_timer.getTime() < MOVE_DURATION:
            if limb != 'REST':
                progress = move_timer.getTime() / MOVE_DURATION
                cursor.pos = (target_pos * progress, 0)
            
            target_left.draw(); target_right.draw(); cursor.draw(); instr_text.draw()
            win.flip()

        win.flip()
        core.wait(ITI_DURATION)

    if run != RUNS_PER_LOOP - 1:
        timer = core.CountdownTimer(BREAK_DURATION)
        while timer.getTime() > 0:
            if 'escape' in event.getKeys(): core.quit()
            break_text.text = f"Break time!\nPlease come back in {int(timer.getTime())}s"
            break_text.draw()
            win.flip()

win.close()
core.quit()