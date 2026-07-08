FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# Prebuilt CPU wheels avoid compiling llama.cpp inside the image.
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Optional: a small GGUF baked at models/local.gguf powers free in-container
# drafting and routing (run scripts/download_model.sh before docker build).
# Without it the agent degrades cleanly to direct remote calls. No downloads
# happen at startup; the 60s readiness window allows none.
COPY models ./models

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are injected by the
# judging harness at run time; nothing secret lives in the image.

ENTRYPOINT ["frugal"]
CMD ["simple"]
