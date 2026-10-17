@echo off
setlocal
cd /d "%~dp0"
call "%~dp0scripts\python_environment.cmd"
"%PYTHON_EXE%" -u -B scripts\run_local_server.py
exit /b %ERRORLEVEL%
