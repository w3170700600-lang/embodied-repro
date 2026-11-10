@echo off
rem Launcher for persistent_trial.py (pure ASCII to avoid cmd codepage issues)
rem huggingface.co is unreachable from this machine; everything needed is cached locally.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
cd /d "%~dp0"
if "%~1"=="" (
  "D:\smolvla-env\Scripts\python.exe" persistent_trial.py --pos fixed --times 5
) else (
  "D:\smolvla-env\Scripts\python.exe" persistent_trial.py %*
)
echo.
echo ---- exited, press any key to close ----
pause >nul
