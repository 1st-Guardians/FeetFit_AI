@echo off
if defined FEETFIT_PYTHON_EXE (
  set "PYTHON_EXE=%FEETFIT_PYTHON_EXE%"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)
