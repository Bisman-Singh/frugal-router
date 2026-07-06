import json

from frugal_router.cli import main


def test_run_command_produces_answers_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    config = tmp_path / "config.yaml"
    config.write_text("router:\n  cache_path: cache.sqlite\n")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "t1", "input": "What is 2 + 2?"}) + "\n")
    output = tmp_path / "answers.jsonl"

    # No local model and no API key: the agent still answers (fallback), never crashes.
    code = main(
        ["--config", str(config), "run", "--tasks", str(tasks), "--output", str(output)]
    )
    assert code == 0
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 1
    answer = json.loads(lines[0])
    assert answer["id"] == "t1"
    assert "answer" in answer
