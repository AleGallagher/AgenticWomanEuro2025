FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
#COPY ./src/.env /app/.env
#COPY ./src/rag/euro2025 /app/src/rag/euro2025
COPY ./src/config/config.json /app/config/config.json
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]