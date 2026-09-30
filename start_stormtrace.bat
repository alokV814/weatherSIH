@echo off
echo =========================================
echo  Starting StormTrace AI System           
echo =========================================

echo Starting FastAPI Backend Engine on port 8000...
cd backend
start "StormTrace Backend" cmd /c "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
cd ..

echo Starting React Frontend on port 5173...
cd frontend
start "StormTrace Frontend" cmd /c "npm run dev -- --host 0.0.0.0 --port 5173"
cd ..

echo =========================================
echo  System Online!                          
echo  Backend: http://localhost:8000          
echo  Frontend: http://localhost:5173         
echo =========================================
echo Close the newly opened terminal windows to stop the services.
pause
