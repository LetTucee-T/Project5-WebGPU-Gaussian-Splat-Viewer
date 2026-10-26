@echo off
cd /d "%~dp0"
call npm run dev -- --host 127.0.0.1 --open
pause
