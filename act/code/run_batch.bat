@echo off
chcp 65001 >nul
setlocal
cd /d D:\lerobot-work

echo.
echo ============================================================
echo   ACT rollout  -  10 trials, success rate
echo ============================================================
echo.
echo   Each round (fully automatic, no key press):
echo     1. The arm resets to the training start pose (3 s).
echo     2. A 10-second countdown starts. Put the green cube at
echo        the centre of the plate DURING the countdown.
echo     3. The policy runs for 11 seconds automatically.
echo        Success is detected automatically (cube lifted 40px+).
echo.
echo   Total: about 4 minutes for 10 rounds.
echo.
echo   Ctrl+C stops it. Finished rounds are still counted.
echo   Do not touch the arm while it moves.
echo.
pause

"D:\lerobot-venv\Scripts\python.exe" batch_rollout.py --n 10 --duration 12 --fps 30 --n-action-steps 100

echo.
echo ========== finished ==========
pause
