@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo AI Proctoring System - Mobile Setup
echo ===================================================

echo [1] Finding your Local IPv4 Address for the Mobile App...
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
    set "userIP=%%i"
    set "userIP=!userIP: =!"
    goto :foundIP
)

:foundIP
if "%userIP%"=="" (
    echo Could not find IP Address. Ensure you are connected to Wi-Fi.
    pause
    exit /b
)

echo Found IP: %userIP%
echo Replacing IP in C:\Production_Mobile\App.js...

:: PowerShell script to definitively replace the SERVER_URL variable without string corruption
powershell -Command "(gc C:\Production_Mobile\App.js) -replace 'const SERVER_URL = ''.*'';', \"const SERVER_URL = 'http://%userIP%:7860';\" | Out-File -encoding ASCII C:\Production_Mobile\App.js"

echo Mobile App successfully bound to http://%userIP%:7860 !

echo.
echo ===================================================
echo [2] BACKEND INSTRUCTIONS
echo ===================================================
echo The mobile app is configured!
echo Please open 'ui/app.py' in your IDE (VSCode, PyCharm, or Anaconda) 
echo and run it manually just like you normally do!
echo.
echo Make sure the Flask server says: "Running on http://0.0.0.0:7860"
echo.

echo ===================================================
echo [3] Starting React Native Mobile Frontend...
echo ===================================================
echo When the QR code appears below, scan it with the 'Expo Go' app on your mobile phone!
cd /d C:\Production_Mobile 
start cmd /k "npm start"
