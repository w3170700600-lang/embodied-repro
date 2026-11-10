@echo off
REM LeRobot 0.6.1 environment launcher (SO-101 ACT reproduction)
REM Usage: double-click this file, then run lerobot-* commands in the window.

call "D:\lerobot-venv\Scripts\activate.bat"
cd /d "D:\lerobot-work"

echo.
echo ==========================================
echo  LeRobot 0.6.1  env: D:\lerobot-venv
echo  Work dir        : D:\lerobot-work
echo ------------------------------------------
echo  python -V
echo  lerobot-find-port
echo  lerobot-find-cameras opencv
echo ==========================================
echo.

cmd /k
