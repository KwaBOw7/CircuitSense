@echo off
REM Windows one-command start: creates a venv, installs deps, serves http://127.0.0.1:8000
cd /d "%~dp0"
if not exist .venv (py -3 -m venv .venv || python -m venv .venv)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if "%PORT%"=="" set PORT=8000
python -m uvicorn circuitsense.api:app --host 127.0.0.1 --port %PORT%
