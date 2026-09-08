"""Portable code-path tests on the development host, not Windows certification."""
from pathlib import Path
from scripts import platform_support
from scripts import workspace
import pytest

def test_interpreter_layouts(monkeypatch):
 root=Path('/test/runtime/venv')
 monkeypatch.setattr(platform_support,'WINDOWS',False)
 assert platform_support.python_in(root)==root/'bin/python'
 monkeypatch.setattr(platform_support,'WINDOWS',True)
 assert platform_support.python_in(root)==root/'Scripts/python.exe'

def test_junctions_are_rejected():
 class Junction:
  def is_symlink(self):return False
  def is_junction(self):return True
 assert platform_support.linked(Junction())

def test_windows_volume_reader_branch(monkeypatch):
 monkeypatch.setattr(workspace,'WINDOWS',True)
 monkeypatch.setattr(workspace,'windows_disk_info',lambda mount:dict(VolumeUUID='WIN-TEST',MountPoint=str(mount)))
 assert workspace.disk_info(Path('/test'))['VolumeUUID']=='WIN-TEST'

def test_sam31_neck_preserves_genuine_scales():
 from backend.sam31 import detector_neck_output
 levels=(object(),object(),object());positions=(object(),object(),object())
 extended,pos=detector_neck_output(None,None,(levels,positions))
 assert extended[:-1]==levels and pos[:-1]==positions
 assert extended[-1] is levels[-1]
 with pytest.raises(RuntimeError):detector_neck_output(None,None,(levels[:2],positions[:2]))
