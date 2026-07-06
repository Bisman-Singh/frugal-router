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

# Mount or bake the GGUF at models/local.gguf (see scripts/download_model.sh).
# FIREWORKS_API_KEY comes from the environment at run time, never from the image.

ENTRYPOINT ["frugal"]
CMD ["--help"]
