FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY python_packages /app/python_packages
ENV PYTHONPATH="/app/python_packages"

COPY app ./app

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.flask_app:app"]