@echo off
REM ============================================================
REM  run.cmd -- run project scripts with the Python that has PyTorch.
REM  (This file is ASCII-only on purpose: Chinese characters in .cmd
REM   files get garbled by cmd.exe and break parsing.)
REM
REM  Why: several Pythons are installed. Typing "python" in cmd may
REM  hit the MSYS2 one (D:\msys64\mingw64\bin\python.exe) which has
REM  no packages installed.
REM
REM  Usage (from this folder):
REM      run 00_env_check.py
REM      run src\step1_numpy_nn.py
REM      run recognize_draw.py
REM ============================================================

set "PY=D:\Program Files\python\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Edit the PY variable in this file to your own Python path.
    pause
    exit /b 1
)

echo [run.cmd] interpreter: %PY%
"%PY%" %*
