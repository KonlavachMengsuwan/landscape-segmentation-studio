@echo off
setlocal
cd /d "%~dp0"
if not exist "..\runtime\venv\Scripts\python.exe" goto missing
"..\runtime\venv\Scripts\python.exe" -B scripts\launch.py start
if errorlevel 1 pause
exit /b
:missing
echo Run Setup Windows.cmd first. Python 3.12 x64 is required.
pause
