FROM python:3.11-slim

WORKDIR /app
ENV DATA_DIR=/data
VOLUME ["/data"]

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY collector.py .

CMD ["python", "-u", "collector.py"]
