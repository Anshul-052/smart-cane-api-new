@echo off
title Starting Sono Lumos Smart Cane...
cd /d C:\Users\VICTUS\smart_cane

:: Activate the virtual environment
call smart_cane_env\Scripts\activate

:: Start FastAPI in a new terminal
start cmd /k "title API Server & uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait a few seconds for the API server to initialize
timeout /t 5

:: Start object detection in a new terminal
start cmd /k "title Object Detection & python detect_objects.py"

echo Project is launching... Please wait.
