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

"""
for port in ports:
    print(port.device)
    print(port.name)
    print(port.description)
    print(port.serial_number)
    print(port.hwid)
    print()
"""
#BT_ADAPTER_VID = 
#BT_ADAPTER_PID = 

# "C:\Users\winni\Desktop\dsi2lsl-win\dsi2lsl.exe --port=COM4"
# may need to change --port, still needs testing

