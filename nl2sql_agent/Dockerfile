# CineAgent (NL2SQL + Agent & Router) - Flask app on port 5050
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=5050

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050
CMD ["python", "app.py"]
