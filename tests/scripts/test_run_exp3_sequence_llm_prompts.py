from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_exp3_sequence_llm_prompts.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_exp3_llm_prompts", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_and_filter_prompt_records(tmp_path):
    module = _load_module()
    prompt_dir = tmp_path / "llm_prompts"
    prompt_dir.mkdir()
    records = [
        {
            "id": "p1",
            "graph_id": "g0",
            "prompt_kind": "oracle_minimax",
            "prompt": "oracle prompt",
        },
        {
            "id": "p2",
            "graph_id": "g1",
            "prompt_kind": "exp3_learned",
            "prompt": "learned prompt",
        },
    ]
    (prompt_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    loaded = module.load_prompt_records(prompt_dir)
    selected = module.select_prompt_records(
        loaded,
        prompt_kind="exp3_learned",
        graph_ids=["g1"],
        limit=1,
    )

    assert [record["id"] for record in loaded] == ["p1", "p2"]
    assert [record["id"] for record in selected] == ["p2"]


def test_provider_payloads_do_not_include_api_keys():
    module = _load_module()

    openai_payload = module.build_openai_compatible_payload(
        "prompt text",
        model="model-x",
        temperature=0.0,
        max_output_tokens=123,
    )
    gemini_payload = module.build_gemini_payload(
        "prompt text",
        temperature=0.1,
        max_output_tokens=456,
    )

    assert openai_payload == {
        "model": "model-x",
        "messages": [{"role": "user", "content": "prompt text"}],
        "temperature": 0.0,
        "max_tokens": 123,
    }
    assert gemini_payload["contents"][0]["parts"] == [{"text": "prompt text"}]
    assert gemini_payload["generationConfig"]["maxOutputTokens"] == 456
    assert "key" not in json.dumps(openai_payload).lower()
    assert "key" not in json.dumps(gemini_payload).lower()


def test_default_output_path_is_under_prompt_responses(tmp_path):
    module = _load_module()

    path = module.default_output_path(
        tmp_path / "llm_prompts",
        provider="ollama",
        models=["qwen2.5:7b"],
    )

    assert path == tmp_path / "llm_prompts" / "responses" / "ollama_qwen2.5_7b.jsonl"


def test_completed_keys_excludes_error_records_so_resume_retries_them(tmp_path):
    module = _load_module()
    out_jsonl = tmp_path / "responses.jsonl"
    records = [
        {"provider": "ollama", "model": "qwen2.5:7b", "prompt_id": "p1", "error": None},
        {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "prompt_id": "p2",
            "error": {"type": "URLError", "message": "connection refused"},
        },
    ]
    out_jsonl.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    completed = module._completed_keys(out_jsonl)

    assert ("ollama", "qwen2.5:7b", "p1") in completed
    assert ("ollama", "qwen2.5:7b", "p2") not in completed
