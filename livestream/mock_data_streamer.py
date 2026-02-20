import uuid # for creating a name
from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL

source_id = "debug-source-id-001"
#fname = sample.data_path() / "sample-ant-raw.fif"
fname = "/Users/chris/Desktop/coding projects/LSL_testing/26_02_04_joint_winston_raw.edf"
player = PlayerLSL(fname, chunk_size=200, source_id=source_id).start()

print("Streaming. source_id =", source_id)
input("Press Enter to stop...\n")
player.stop()