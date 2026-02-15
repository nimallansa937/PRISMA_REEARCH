@echo off
title PRISMA Research Agent - Starting...
color 0A

echo.
echo  ============================================
echo   PRISMA SYSTEMATIC REVIEW ENGINE v3.0
echo   22 Multi-Agent Research System
echo  ============================================
echo.

:: Set working directory
cd /d "C:\Users\chari\OneDrive\Documents\academic-research-agent"

:: Check if Ollama is running
echo  [1/3] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Ollama not running. Starting Ollama...
    start "" "ollama" serve
    timeout /t 3 /nobreak >nul
) else (
    echo  [OK] Ollama is running
)

:: Kill any existing Streamlit on port 8503
echo  [2/3] Preparing dashboard...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8503 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: Launch Streamlit
echo  [3/3] Launching dashboard...
echo.
echo  ============================================
echo   Dashboard: http://localhost:8503
echo   Press Ctrl+C to stop
echo  ============================================
echo.

:: Open browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8503"

:: Run Streamlit (this blocks until closed)
"venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.port 8503 --server.headless true --browser.gatherUsageStats false

echo.
echo  Agent stopped. Press any key to close.
pause >nul
