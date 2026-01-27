FROM semtech/mu-python-template:feature-fastapi
# Or use python:3.11-slim if you prefer a standard image
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "80"]

# The semtech template usually expects the app at /app and handles requirements.
# If using the template, we might rely on its ONBUILD triggers or set up manually.
# For safety, we ensure requirements are installed:

# If the base image doesn't auto-install requirements:
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app
