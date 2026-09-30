#!/bin/bash
echo "========================================="
echo " Starting StormTrace AI System           "
echo "========================================="

# Start backend
echo "Starting FastAPI Backend Engine on port 8000..."
cd backend
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Start frontend
echo "Starting React Frontend on port 5173..."
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!
cd ..

echo "========================================="
echo " System Online!                          "
echo " Backend: http://localhost:8000          "
echo " Frontend: http://localhost:5173         "
echo "========================================="
echo "Press Ctrl+C to gracefully stop all services."

# Trap Ctrl+C (SIGINT) to kill background processes
trap "echo -e '\nStopping StormTrace AI...'; kill $BACKEND_PID $FRONTEND_PID; exit 0" SIGINT SIGTERM

# Wait indefinitely for processes
wait $BACKEND_PID $FRONTEND_PID
