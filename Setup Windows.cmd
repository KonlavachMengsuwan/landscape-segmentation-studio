@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if errorlevel 1 goto python_missing
py -3.12 -B scripts\setup.py --initialize-ssd --download-models %*
if errorlevel 1 goto fail
"..\runtime\venv\Scripts\python.exe" -B scripts\verify_models.py --model baseline
if errorlevel 1 goto fail
echo Setup and real local model checks passed. Open Start Studio.cmd.
pause
exit /b 0
:python_missing
echo Python 3.12 with the py launcher is required. Other Python versions may stay installed.
echo See START_HERE.md for the official Python installation steps, then run setup again.
goto fail
:fail
echo Setup stopped. Read the error above. See docs\WINDOWS_SETUP.md.
pause
exit /b 1
