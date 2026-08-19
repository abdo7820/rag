FROM python:3.11-slim

WORKDIR /app

COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

COPY . .

CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
