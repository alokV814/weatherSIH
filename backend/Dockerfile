FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file from the inner backend directory
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all backend code
COPY . .

EXPOSE 7860 8000

ENV PYTHONPATH=/app:/app/backend

CMD ["sh", "-c", "python -m uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
