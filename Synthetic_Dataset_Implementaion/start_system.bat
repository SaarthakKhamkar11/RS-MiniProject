@echo off
echo ========================================
echo Restaurant Recommendation System
echo ========================================
echo.

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Starting the system...
echo Backend will run on: http://localhost:5000
echo Frontend will run on: http://localhost:8000
echo.

start "Backend Server" cmd /k "python app.py"
timeout /t 3 /nobreak >nul
start "Frontend Server" cmd /k "python static_server.py"

echo.
echo System started successfully!
echo Please wait for both servers to start, then open your browser.
echo.
pause

