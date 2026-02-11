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

TRIG_LEFT = 1
TRIG_RIGHT = 2
TRIG_REST = 3

# This block allows the script to run WITHOUT an EEG box
try:
    mmbts = serial.Serial(PORT, baudrate=115200, timeout=1)
    print(f"SUCCESS: Connected to EEG box on {PORT}")
except:
    mmbts = None
    print("OFFLINE MODE: No EEG box found. Triggers will not be sent, but task will run.")

win = visual.Window(color="gray", units='norm', fullscr=True)

target_left = visual.Circle(win, radius=0.15, lineColor='black', lineWidth=3, pos=(-0.6, 0))
target_right = visual.Circle(win, radius=0.15, lineColor='black', lineWidth=3, pos=(0.6, 0))

instr_text = visual.TextStim(win, text="", pos=(0, 0.6), color="black", height=0.08, wrapWidth=1.8)
break_text = visual.TextStim(win, text="", color="black", height=0.1)

for run in range(RUNS_PER_LOOP):
    for trial in range(NUM_TRIALS):
        if 'escape' in event.getKeys(): core.quit()

        direction = random.choice(['LEFT', 'RIGHT', 'REST'])
        
        if direction == 'LEFT': side_trigger = TRIG_LEFT
        elif direction == 'RIGHT': side_trigger = TRIG_RIGHT
        else: side_trigger = TRIG_REST
        
        target_obj = None
        if direction == 'LEFT': target_obj = target_left
        elif direction == 'RIGHT': target_obj = target_right
        
        if direction == 'LEFT': target_pos = -0.6
        elif direction == 'RIGHT': target_pos = 0.6

        target_left.fillColor = target_right.fillColor = None
        
        if direction == 'REST':
            instr_text.text = "Resting: Keep your arms still."
        else:
            instr_text.text = f"Prepare to REACH with your dominant arm toward the {direction.upper()} target."
        
        target_left.draw(); target_right.draw(); instr_text.draw()
        win.flip()
        core.wait(PREP_DURATION)

        if direction != 'REST':
            target_obj.fillColor = 'green'
            instr_text.text = f"IMAGINE REACHING"
        
        # This only executes if mmbts was successfully connected
        if mmbts:
            win.callOnFlip(mmbts.write, bytes([side_trigger]))
        
        target_left.draw(); target_right.draw(); instr_text.draw()
        win.flip()
        core.wait(FLASH_ONLY_DURATION)

        move_timer = core.Clock()
        core.wait(MOVE_DURATION)
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