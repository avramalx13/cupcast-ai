FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .

EXPOSE 8000
