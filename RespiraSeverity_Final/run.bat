@echo off
REM Usage: run.bat <script.py> [args]
REM Runs a Python script from the project venv, writing output to both console and a log file.
set VENV_PY=%~dp0.venv\Scripts\python.exe
set SCRIPT=%~dp0%1
set LOG=%~dp0%~n1.log
echo Running: %SCRIPT%
echo Log: %LOG%
"%VENV_PY%" "%SCRIPT%" %2 %3 %4 > "%LOG%" 2>&1
echo Exit code: %ERRORLEVEL%
