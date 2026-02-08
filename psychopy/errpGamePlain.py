from psychopy import visual, core, event, gui
import numpy as np
import time
import serial

# --- Configuration ---
WIN_SIZE = [1200, 800]
CURSOR_SIZE = 15
TARGET_SIZE = 40

# Movement Settings
STEP_SIZE = 60              # Distance of one "move"
TARGET_DISTANCE = 5 * STEP_SIZE # Target is 5 steps away
MOVE_DURATION = 1.25         # Seconds it takes to glide to next point (Continuous movement)

# Error Parameters
ERROR_PROB = 0.15           # 30% chance
ERROR_MAGNITUDES = [90, 180, 270] 

# Experiment Settings
N_TRIALS = 100

PORT = 'COM6'
ERROR_90_TRIGGER = 1 # signifies an error trial
ERROR_180_TRIGGER = 2 # signifies an error trial
ERROR_270_TRIGGER = 3 # signifies an error trial
NON_ERROR_TRIGGER = 4 # signifies a non-error trial

mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()

# --- Setup ---
win = visual.Window(size=WIN_SIZE, units='pix', color='black')

cursor = visual.Circle(win, radius=CURSOR_SIZE, fillColor='blue', lineColor='white')
target = visual.Rect(win, width=TARGET_SIZE, height=TARGET_SIZE, fillColor='red', lineColor='white')

# Simple Instruction Text
instr = visual.TextStim(win, text="", color='white', height=30, pos=(0, 350))

# --- Helper Functions ---
def get_movement_vector(keys, step):
    dx, dy = 0, 0
    if 'up' in keys: dy = step
    elif 'down' in keys: dy = -step
    elif 'left' in keys: dx = -step
    elif 'right' in keys: dx = step
    return np.array([dx, dy], dtype=float)

def rotate_vector(vec, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    new_x = vec[0] * c - vec[1] * s
    new_y = vec[0] * s + vec[1] * c
    return np.array([new_x, new_y])

# --- Experiment Loop ---
start_msg = visual.TextStim(win, text="Reach the Red Square.\n\nPress arrow keys to move.\n\nPress any key to start.", color='white')
start_msg.draw()
win.flip()
event.waitKeys()

for trial_i in range(N_TRIALS):
    
    # 1. Reset Trial
    cursor.pos = (0, 0)
    
    # Position Target
    target_angle = np.random.uniform(0, 2*np.pi)
    target.pos = (TARGET_DISTANCE * np.cos(target_angle), TARGET_DISTANCE * np.sin(target_angle))
    
    # Log Start
    trial_start_time = time.time()

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
            
        if keys:
            # --- Calculate Destination (Math) ---
            start_pos = np.array(cursor.pos)
            move_vec = get_movement_vector(keys, STEP_SIZE)
            
            # Error Logic
            is_error = False
            rotation = 0
            if np.random.random() < ERROR_PROB:
                is_error = True
                rotation = np.random.choice(ERROR_MAGNITUDES)
                if np.random.random() > 0.5: rotation = -rotation
                move_vec = rotate_vector(move_vec, rotation)
                
                ERROR_TRIGGER = 0
                if rotation == 90:
                    ERROR_TRIGGER = ERROR_90_TRIGGER
                elif rotation == 180:
                    ERROR_TRIGGER = ERROR_180_TRIGGER
                else:
                    ERROR_TRIGGER = ERROR_270_TRIGGER
                win.callOnFlip(mmbts.write, bytes([ERROR_TRIGGER]))
            else:
                win.callOnFlip(mmbts.write, bytes([NON_ERROR_TRIGGER]))

            end_pos = start_pos + move_vec
            
            # Boundary Clamp (Prevent moving off screen)
            end_pos[0] = np.clip(end_pos[0], -WIN_SIZE[0]//2, WIN_SIZE[0]//2)
            end_pos[1] = np.clip(end_pos[1], -WIN_SIZE[1]//2, WIN_SIZE[1]//2)

            # --- Animate Movement (Visual) ---
            animation_clock = core.Clock()
            while animation_clock.getTime() < MOVE_DURATION:
                # Calculate progress (0.0 to 1.0)
                t = animation_clock.getTime() / MOVE_DURATION
                
                # Linear Interpolation (Lerp)
                current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
                current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * t
                cursor.pos = (current_x, current_y)
                
                target.draw()
                cursor.draw()
                instr.draw()
                win.flip()
            
            # Ensure final position is exact
            cursor.pos = end_pos
            
            # --- Check Success ---
            dist = np.linalg.norm(cursor.pos - target.pos)
            if dist < (CURSOR_SIZE + TARGET_SIZE):
                trial_complete = True
                
                # Success Feedback
                feedback = visual.TextStim(win, text="Target Reached!", color='green', height=40)
                for _ in range(60): # Show for ~1 sec (60 frames)
                    target.draw()
                    cursor.draw()
                    feedback.draw()
                    win.flip()

# --- End ---
win.close()