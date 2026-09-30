# BuddySync ML service. The trained artifacts/ and the data/ CSVs are in the repo and copied in.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Render (and most PaaS) set $PORT. One process by default: with the face + liveness models loaded it uses
# ~430 MB; raise WEB_CONCURRENCY only on instances with ≥ 1 GB per extra process.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
