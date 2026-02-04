from psychopy import visual, core, event
import numpy as np
import serial

# --- Configuration ---
NUM_TRIALS = 2
STATIC_DURATION = 2.0   # seconds (No movement)
MOVE_DURATION = 33.0    # seconds (Movement)
TRIAL_DURATION = STATIC_DURATION + MOVE_DURATION
BREAK_DURATION = 10.0    # seconds
SCREEN_BG_COLOR = 'black'
TARGET_COLOR = 'white'
REFRESH_RATE = 60

# --- Triggers ---
PORT = 'COM3'
MOVEMENT_START_STOP = 1
CHANGE_DIR_LEFT = 2
CHANGE_DIR_RIGHT = 3

mmbts = serial.Serial()
mmbts.port = PORT
mmbts.open()

# Window setup
win = visual.Window(
    size=[1024, 768],
    fullscr=False,
    screen=0,
    color=SCREEN_BG_COLOR,
    units='height'
)

# Visual Elements
target_stim = visual.Circle(
    win, 
    radius=0.03, 
    fillColor=TARGET_COLOR, 
    lineColor=TARGET_COLOR
)

instruction_text = visual.TextStim(win, text="", color='white')

# --- Data Logging ---
all_trial_events = []

def generate_delayed_trajectory(static_dur, move_dur, fps):
    """
    Generates a trajectory that stays static for `static_dur` and then 
    moves for `move_dur` using the 1D Left/Right sine wave logic.
    """
    # 1. Generate the Movement Phase
    n_frames_move = int(move_dur * fps)
    t_move = np.linspace(0, move_dur, n_frames_move)
    
    # Slower frequencies (from previous request)
    freqs_x = [0.05, 0.11, 0.23]
    
    x_move = np.zeros_like(t_move)
    for f in freqs_x:
        # Random phase shift for uniqueness
        x_move += np.sin(2 * np.pi * f * t_move + np.random.rand() * 2 * np.pi)
        
    # Normalize width
    x_move = (x_move / np.max(np.abs(x_move))) * 0.4
    
    # 2. Generate the Static Phase
    # The target should hold the position where the movement STARTS
    # to avoid a visual "jump" at 3s.
    start_pos = x_move[0]
    n_frames_static = int(static_dur * fps)
    x_static = np.full(n_frames_static, start_pos)
    
    # 3. Combine
    path_x = np.concatenate([x_static, x_move])
    path_y = np.zeros_like(path_x) # 1D restriction (y=0)
    
    return path_x, path_y

# --- Experiment Start ---
instruction_text.text = f"Press space to start"
instruction_text.draw()
win.callOnFlip(mmbts.write, bytes([MOVEMENT_START_STOP]))
win.flip()
event.waitKeys(keyList=['space'])

quit_experiment = False

# --- Main Experiment Loop ---
for trial_idx in range(1, NUM_TRIALS + 1):
    
    # 1. Generate path with static prefix
    path_x, path_y = generate_delayed_trajectory(STATIC_DURATION, MOVE_DURATION, REFRESH_RATE)
    
    # 2. Setup Trial State
    clock = core.Clock()
    frame_idx = 0
    current_direction = 'Static'
    prev_x = path_x[0] # Initialize with starting position

    # 3. Run Trial Loop
    while clock.getTime() < TRIAL_DURATION and frame_idx < len(path_x):
        # Update Position
        target_x = path_x[frame_idx]
        target_y = path_y[frame_idx]
        target_stim.pos = (target_x, target_y)
        
        # Detect Movement / Direction Change
        # We compare current target position to the previous frame's position
        dx = target_x - prev_x
        
        if dx > 1e-9:
            frame_direction = 'Right'
        elif dx < -1e-9:
            frame_direction = 'Left'
        else:
            frame_direction = 'Static'
        
        trigger_to_send = 0
        # Trigger logic

        if frame_direction != current_direction:
            if frame_direction == 'Static':
                trigger_to_send = MOVEMENT_START_STOP
            elif frame_direction == 'Right':
                trigger_to_send = CHANGE_DIR_RIGHT
            elif frame_direction == 'Left':
                trigger_to_send = CHANGE_DIR_LEFT
            
            current_direction = frame_direction
        
        # Update for next loop
        prev_x = target_x
        
        # Render
        target_stim.draw()
        if trigger_to_send != 0:
            win.callOnFlip(mmbts.write, bytes([trigger_to_send]))
        win.flip()
        
        frame_idx += 1
        
        if 'escape' in event.getKeys():
            quit_experiment = True
            break
    
    if quit_experiment:
        break
        
    # 4. Break between trials
    if trial_idx < NUM_TRIALS:
        break_timer = core.Clock()
        while break_timer.getTime() < BREAK_DURATION:
            remaining = int(np.ceil(BREAK_DURATION - break_timer.getTime()))
            instruction_text.text = f"Trial {trial_idx} Complete\nBreak: {remaining} s"
            instruction_text.draw()
            win.flip()
            
            if 'escape' in event.getKeys():
                quit_experiment = True
                break
        if quit_experiment:
            break

win.close()

# --- Output Results ---
print(f"{'Trial':<6} | {'Time (s)':<10} | {'Event':<20} | {'Direction'}")
print("-" * 55)
for log in all_trial_events:
    # Filter out initial start logs if you only want movement specific ones
    print(f"{log['trial']:<6} | {log['time']:<10.4f} | {log['event']:<20} | {log['direction']}")

core.quit()