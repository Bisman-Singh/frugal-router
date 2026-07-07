FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

# Prebuilt CPU wheels avoid compiling llama.cpp inside the image.
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# The GGUF must be baked into the image before submission: the judging harness
# gives the container no network time for downloads inside its 60s readiness
# window. Run scripts/download_model.sh before docker build.
COPY models ./models

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are injected by the
# judging harness at run time; nothing secret lives in the image.

ENTRYPOINT ["frugal"]
CMD ["harness"]
