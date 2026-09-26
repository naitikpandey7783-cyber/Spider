# The Daily Bugle Truth Serum — API container
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Train the models at build time so the container is ready to serve
# immediately (skip this line and mount a volume instead if you'd rather
# train at runtime or reuse artifacts built elsewhere).
RUN python data/generate_dataset.py && python models/train_ensemble.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
