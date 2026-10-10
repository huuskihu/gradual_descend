@echo off
REM ============================================================
REM  fix_path.cmd -- one-click fix: make "python" in cmd point to
REM  D:\Program Files\python\python.exe (the one that has PyTorch)
REM  instead of the MSYS2 python.
REM
REM  ASCII-only on purpose (cmd.exe mangles Chinese in .cmd files).
REM  How to use: just double-click this file, then click "Yes" on
REM  the UAC prompt.
REM ============================================================

fltmc >nul 2>&1
if %errorlevel% NEQ 0 (
    echo Requesting administrator permission - click YES on the UAC prompt...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo Running with administrator rights...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_fix_path.ps1"
echo.
echo === Result (also saved to outputs\_path_fix_result.txt) ===
type "%~dp0outputs\_path_fix_result.txt"
echo.
echo Now open a NEW cmd window and type:  python
pause
