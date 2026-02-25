import serial.tools.list_ports

TRIGGER_HUB_VID = 0x2341
TRIGGER_HUB_PID = 0x8037

ports = serial.tools.list_ports.comports()

def find_port_by_vid_pid(vid, pid):
    for port in ports:
        if port.vid == vid and port.pid == pid:
            return port.device
    return None


found = find_port_by_vid_pid(TRIGGER_HUB_VID, TRIGGER_HUB_PID)
print(found)


for port in ports:
    print(port.device)
    print(port.name)
    print(port.description)
    print(port.serial_number)
    print(port.hwid)
    print()

#BT_ADAPTER_VID = 
#BT_ADAPTER_PID = 

# "C:\Users\winni\Desktop\dsi2lsl-win\dsi2lsl.exe --port=COM4"
# may need to change --port, still needs testing

"""
COM6
COM6
COM6
USB Serial Device (COM6)
5&1E62F63C&0&4
USB VID:PID=2341:8037 SER=5&1E62F63C&0&4 LOCATION=1-4:x.0

COM3
COM3
Standard Serial over Bluetooth link (COM3)
None
BTHENUM\{00001101-0000-1000-8000-00805F9B34FB}_VID&00010047_PID&F000\7&48F7B1C&0&A46DD47E2451_C00000000

COM4
COM4
Standard Serial over Bluetooth link (COM4)
None
BTHENUM\{00001101-0000-1000-8000-00805F9B34FB}_LOCALMFG&0000\7&48F7B1C&0&000000000000_00000026
"""
