@echo off
title Mika Web App
cd /d "%~dp0"

:: Kill anything already on port 8003
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8003 ^| findstr LISTENING') do (
    echo Killing old process %%a on port 8003...
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

call C:\Users\Hunte\Downloads\myai\.venv\Scripts\activate
pip install fastapi uvicorn python-multipart -q
python server.py
pause
