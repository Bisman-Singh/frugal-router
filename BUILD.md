# v47-final — the shipped image

Image: docker.io/bismansinghmadaan/frugal-router:v47-final

Contents: v40-onv29's exact source + terse prompts (simple.py CONTRACTS) +
deterministic solvers (facts.py, logic_search.py, math extras in solvers.py).
Excludes the tuned sentiment/summary heuristics and heavy local lanes.

Build (legacy builder → produces v40-byte-identical layering on v29):

    DOCKER_BUILDKIT=0 docker build -f Dockerfile -t bismansinghmadaan/frugal-router:v47-final .
    docker push bismansinghmadaan/frugal-router:v47-final

Fallback image (guaranteed gate-passer, rank 83): bismansinghmadaan/frugal-router:v40-onv29
