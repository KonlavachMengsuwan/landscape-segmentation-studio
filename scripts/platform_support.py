"""Local volume and process primitives; no registry, account, or network access."""
import os
from pathlib import Path
import subprocess

NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
WINDOWS = os.name == 'nt'

def linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())

def python_in(venv):
    return venv / ('Scripts/python.exe' if WINDOWS else 'bin/python')

def windows_disk_info(mount):
    import ctypes
    from ctypes import wintypes
    volume = str(mount)
    if not volume.endswith('\\'): volume += '\\'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kind = kernel.GetDriveTypeW(ctypes.c_wchar_p(volume))
    if kind not in (2, 3):
        raise RuntimeError('Choose a dedicated folder on a connected local drive; network drives are unsupported.')
    serial, maximum, flags = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
    label, filesystem = ctypes.create_unicode_buffer(261), ctypes.create_unicode_buffer(261)
    if not kernel.GetVolumeInformationW(ctypes.c_wchar_p(volume), label, len(label), ctypes.byref(serial), ctypes.byref(maximum), ctypes.byref(flags), filesystem, len(filesystem)):
        raise OSError(ctypes.get_last_error(), 'Cannot verify the connected project volume')
    return dict(VolumeUUID=f'WIN-{serial.value:08X}', MountPoint=str(mount), Mounted=True,
                WritableVolume=not bool(flags.value & 0x80000), FilesystemType=filesystem.value.lower(), Internal=kind==3)

def process_command(pid):
    if WINDOWS:
        # PID is validated by the caller; no user text is evaluated as PowerShell.
        result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
            f'(Get-CimInstance Win32_Process -Filter "ProcessId = {int(pid)}").CommandLine'],
            capture_output=True,text=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        result = subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=5)
    return result.returncode, result.stdout.strip()
