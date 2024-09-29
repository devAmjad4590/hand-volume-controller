import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import threading


def get_system_volume_controller():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume
volume = get_system_volume_controller().GetMasterVolumeLevelScalar();

def reduce_volume_gradually(volume_controller, target_volume, step, up):
        current_volume = volume_controller.GetMasterVolumeLevelScalar()
        setPrint(current_volume)
        
        if up:
            current_volume = min(target_volume, current_volume + step)
        else:
            current_volume = max(target_volume, current_volume - step)
        
        volume_controller.SetMasterVolumeLevelScalar(current_volume, None)

def setPrint(current_volume):
    global volume
    volume = current_volume

def getPrint():
    return volume

if __name__ == "__main__":
    volume_controller = get_system_volume_controller()
    reduce_volume_gradually(volume_controller, target_volume=0.0, step=0.05, up=True)
    getPrint()