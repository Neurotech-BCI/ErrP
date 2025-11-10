import numpy as np

def csv_to_np(path):
    data = np.genfromtxt(path, delimiter = '\t', invalid_raise=False)
    num_cols = data[0,:].shape[0]
    filtered_data = np.array([row for row in data if len(row) == num_cols])
    eeg_data = filtered_data[:, 0:17]
    reset_indices = np.where(eeg_data[:,0] == 0)[0]
    subarrays = [eeg_data[start:start + 128, 1:] for start in reset_indices]
    target_shape = subarrays[0].shape
    for i in range(len(subarrays)):
        if subarrays[i].shape != target_shape:
            del subarrays[i]
    result_3d = np.array(subarrays)
    result = np.transpose(result_3d, (0, 2, 1))
    return result