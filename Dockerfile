FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-serving.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serving.txt
COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY models/deploy ./models/deploy
ENV DERMAAI_BUNDLE=/app/models/deploy
EXPOSE 8000 8501
# API:        docker run -p 8000:8000 dermaai
# Dashboard:  docker run -p 8501:8501 dermaai streamlit run dashboard/app.py --server.address 0.0.0.0
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
