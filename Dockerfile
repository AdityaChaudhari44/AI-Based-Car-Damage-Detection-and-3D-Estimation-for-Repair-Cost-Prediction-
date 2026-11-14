FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements*.txt ./
# Heavier image with PyTorch MiDaS:  docker build --build-arg EXTRAS="-r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu" .
ARG EXTRAS=""
RUN pip install --no-cache-dir -r requirements.txt ${EXTRAS}
COPY car_damage ./car_damage
RUN useradd -m app && chown -R app /app
USER app
ENV DEPTH_BACKEND=auto
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health')"
CMD ["sh", "-c", "uvicorn car_damage.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
