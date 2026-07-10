#!/usr/bin/env bash
# Real-image smoke test: build the tag, run the ACTUAL container against a
# scripted mock endpoint, and assert the contract end-to-end. This is the test
# the in-repo pytest suite cannot do (stale tag, COPY omission, dependency
# mismatch, wrong entrypoint). Run before pushing any tag:
#
#   ./scripts/smoke_image.sh bismansinghmadaan/frugal-router:vNN
set -euo pipefail

TAG="${1:?usage: smoke_image.sh <image:tag>}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"; kill "${MOCK_PID:-0}" 2>/dev/null || true' EXIT

echo "== build =="
docker buildx build --platform linux/amd64 -t "$TAG" --load . >/dev/null

echo "== mock endpoint =="
python3 - "$DIR" <<'PY' &
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"].lower()
        if "sentiment" in user:
            text = "Positive. The reviewer praises the product."
        elif "change" in user or "%" in user:
            text = "7x3=21, 50-21=29.\nAnswer: 29"
        else:
            text = "A concise, correct factual explanation of the topic."
        payload = {"id": "m", "object": "chat.completion", "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a): pass

srv = HTTPServer(("0.0.0.0", 8971), H)
open(sys.argv[1] + "/mock.ready", "w").write("ok")
srv.serve_forever()
PY
MOCK_PID=$!
for _ in $(seq 1 20); do [ -f "$DIR/mock.ready" ] && break; sleep 0.5; done

echo "== run container =="
mkdir -p "$DIR/in" "$DIR/out"
cat > "$DIR/in/tasks.json" <<'JSON'
[{"task_id":"t1","prompt":"Classify the sentiment of: 'Great product!'"},
 {"task_id":"t2","prompt":"Pens cost $3. Buying 7 with a $50 note, how much change?"},
 {"task_id":"t3","prompt":"What is 15% of 240?"}]
JSON
docker run --rm --add-host host.docker.internal:host-gateway \
  -e FIREWORKS_API_KEY=smoke -e ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it" \
  -e FIREWORKS_BASE_URL="http://host.docker.internal:8971/v1" \
  -v "$DIR/in":/input:ro -v "$DIR/out":/output "$TAG"

echo "== assert =="
python3 - "$DIR/out" <<'PY'
import json, sys
out = sys.argv[1]
rs = {r["task_id"]: r["answer"] for r in json.load(open(out + "/results.json"))}
assert len(rs) == 3 and all(v.strip() for v in rs.values()), rs
assert rs["t3"] == "36", "solver task must be answered exactly, zero tokens"
log = json.load(open(out + "/inference_log.json"))
assert log["summary"]["solver_answered"] >= 1
print("IMAGE SMOKE PASS:", json.dumps(log["summary"]))
PY
