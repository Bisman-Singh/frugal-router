FROM bismansinghmadaan/frugal-router:v29
COPY pyproject.toml README.md ./
COPY src /app/src
RUN pip install --no-cache-dir --no-deps --force-reinstall . && \
    python -c "import frugal_router.simple; print('import OK')"
ENV LOCAL=1 LOCAL_MODEL_PATH=/app/models/local.gguf LOCAL_THREADS=2 LOCAL_SUFFIX="" CONFIRM=off LOCAL_TIME_BUDGET=360 BATCH=0
