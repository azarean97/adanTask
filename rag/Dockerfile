# Medical RAG (KDIGO CKD) - Flask app on port 5001
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=5001

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["python", "app.py"]
