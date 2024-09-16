@echo off
REM
if "%~1"=="" (
    echo "Usage: build.bat <filename.cpp>"
    exit /b 1
)

REM
set filename=%~n1

REM
g++ "%~1" -std=c++23 -O3 -o "%filename%.exe"

REM
if "%errorlevel%"=="0" (
    echo "Compilation successful. Output: %filename%.exe"
) else (
    echo "Compilation failed."
)
