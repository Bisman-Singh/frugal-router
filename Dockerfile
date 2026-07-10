FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# The scored path is remote-only: no local weights, no llama.cpp, no ML extras.
# Anything the entrypoint never imports stays out of the image so every judge
# pull is fast and the build is reproducible against the pinned deps.
RUN pip install --no-cache-dir .

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are injected by the
# judging harness at run time; nothing secret lives in the image.

ENTRYPOINT ["frugal"]
CMD ["simple"]
