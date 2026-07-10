from __future__ import annotations

import json

from sequence_game.experiments.exp3_sequence.llm_prompt_packaging import (
    package_prompts,
)


def test_package_prompts_writes_per_graph_jsonl_and_text(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "dt").mkdir(parents=True)
    _write_minimal_run(run_dir)

    manifest = package_prompts(
        run_dir,
        out_dir=tmp_path / "prompts",
        top_actions=1,
        top_routes=1,
    )

    assert manifest["prompt_count"] == 2
    jsonl_path = tmp_path / "prompts" / "prompts.jsonl"
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["graph_id"] for record in records] == ["g_bottle", "g_parallel"]
    assert records[0]["messages"] == [
        {"role": "user", "content": records[0]["prompt"]}
    ]
    assert "bottleneck_node_count > 0.500" in records[0]["prompt"]
    assert "oracle_retention target value: 0" in records[0]["prompt"]
    assert "action_id=edge_intercept_resend:a-s" in records[0]["prompt"]
    assert "route_id=r0" in records[0]["prompt"]
    assert (tmp_path / "prompts" / "prompts" / "g_bottle.txt").exists()
    assert (tmp_path / "prompts" / "manifest.json").exists()


def test_package_prompts_separates_oracle_and_learned_exp3_terms(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "dt").mkdir(parents=True)
    _write_minimal_run(run_dir)

    manifest = package_prompts(
        run_dir,
        out_dir=tmp_path / "prompts",
        top_actions=2,
        top_routes=2,
        limit=1,
        prompt_kind="both",
    )

    assert manifest["prompt_count"] == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "prompts" / "prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_kind = {record["prompt_kind"]: record["prompt"] for record in records}
    oracle_prompt = by_kind["oracle_minimax"]
    learned_prompt = by_kind["exp3_learned"]

    assert "This is an oracle/minimax interpretation prompt" in oracle_prompt
    assert "complete-information minimax oracle" in oracle_prompt
    assert "This is a learned-strategy Exp3 interpretation prompt" in learned_prompt
    assert "learned time-averaged Exp3 strategy" in learned_prompt
    assert "complete-information minimax oracle" not in learned_prompt
    assert "oracle_eve_strategy_prob" not in learned_prompt
    assert "oracle_alice_strategy_prob" not in learned_prompt
    assert "When a direct strategy probability and a DT leaf prediction disagree" in learned_prompt
    assert "row_role=selected/high-probability learned Eve action" in learned_prompt
    assert "row_role=diagnostic comparison action" in learned_prompt
    assert "row_role=selected/high-probability learned Alice route" in learned_prompt
    assert "row_role=diagnostic comparison route" in learned_prompt
    assert "weak/low-fidelity" in learned_prompt
    assert "toy" not in learned_prompt.lower()


def test_package_prompts_limit_for_smoke_runs(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "dt").mkdir(parents=True)
    _write_minimal_run(run_dir)

    manifest = package_prompts(run_dir, out_dir=tmp_path / "prompts", limit=1)

    assert manifest["prompt_count"] == 1
    lines = (tmp_path / "prompts" / "prompts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1


def _write_minimal_run(run_dir):
    corpus = {
        "summary": {
            "graph_count": 2,
            "families": {"single_bottleneck": 1, "disjoint_parallel": 1},
        },
        "graphs": [
            _graph("g_bottle", "single_bottleneck", ["a", "s", "b"]),
            _graph("g_parallel", "disjoint_parallel", ["a", "r0", "b"]),
        ],
    }
    dt_payload = {
        "graph_rows": [
            {
                "graph_id": "g_bottle",
                "family": "single_bottleneck",
                "features": [1.0, 1.0],
                "oracle_retention": 0.0,
                "oracle_value": 0.0,
                "baseline_rate": 1.0,
                "exp3_final_retention": 0.0,
                "exp3_exploitability_vs_oracle": 0.0,
            },
            {
                "graph_id": "g_parallel",
                "family": "disjoint_parallel",
                "features": [0.0, 2.0],
                "oracle_retention": 0.5,
                "oracle_value": 0.5,
                "baseline_rate": 1.0,
                "exp3_final_retention": 0.45,
                "exp3_exploitability_vs_oracle": 0.1,
            },
        ],
        "action_rows": [
            {
                "graph_id": "g_bottle",
                "family": "single_bottleneck",
                "action_id": "edge_intercept_resend:a-s",
                "features": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "oracle_eve_strategy_prob": 1.0,
                "exp3_eve_empirical_prob": 0.75,
                "action_expected_denial_under_oracle_alice": 1.0,
                "active_hit_sample_count": 4,
                "active_hit_accepted_rate": 0.0,
                "active_hit_mean_chsh_s": 0.1,
                "active_hit_mean_qber": 0.5,
            },
            {
                "graph_id": "g_bottle",
                "family": "single_bottleneck",
                "action_id": "no_attack",
                "features": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                "oracle_eve_strategy_prob": 0.0,
                "exp3_eve_empirical_prob": 0.0,
                "action_expected_denial_under_oracle_alice": 0.0,
                "active_hit_sample_count": 0,
                "active_hit_accepted_rate": None,
                "active_hit_mean_chsh_s": None,
                "active_hit_mean_qber": None,
            },
            {
                "graph_id": "g_parallel",
                "family": "disjoint_parallel",
                "action_id": "memory_degradation:r0",
                "features": [0.0, 0.0, 1.0, 0.5, 0.0, 2.0, 2.0],
                "oracle_eve_strategy_prob": 0.5,
                "exp3_eve_empirical_prob": 0.6,
                "action_expected_denial_under_oracle_alice": 0.5,
                "active_hit_sample_count": 4,
                "active_hit_accepted_rate": 0.0,
                "active_hit_mean_chsh_s": 1.2,
                "active_hit_mean_qber": 0.25,
            },
            {
                "graph_id": "g_parallel",
                "family": "disjoint_parallel",
                "action_id": "no_attack",
                "features": [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                "oracle_eve_strategy_prob": 0.0,
                "exp3_eve_empirical_prob": 0.0,
                "action_expected_denial_under_oracle_alice": 0.0,
                "active_hit_sample_count": 0,
                "active_hit_accepted_rate": None,
                "active_hit_mean_chsh_s": None,
                "active_hit_mean_qber": None,
            },
        ],
        "route_rows": [
            {
                "graph_id": "g_bottle",
                "family": "single_bottleneck",
                "route_id": "r0",
                "features": [2.0, 800.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "oracle_alice_strategy_prob": 1.0,
                "exp3_alice_empirical_prob": 0.8,
                "no_attack_key_rate": 1.0,
                "worst_case_key_rate": 0.0,
            },
            {
                "graph_id": "g_bottle",
                "family": "single_bottleneck",
                "route_id": "r_diag",
                "features": [2.0, 850.0, 1.0625, 1.0, 1.0, 1.0, 0.0],
                "oracle_alice_strategy_prob": 0.0,
                "exp3_alice_empirical_prob": 0.0,
                "no_attack_key_rate": 0.9,
                "worst_case_key_rate": 0.0,
            },
            {
                "graph_id": "g_parallel",
                "family": "disjoint_parallel",
                "route_id": "r0",
                "features": [2.0, 800.0, 1.0, 1.0, 0.0, 2.0, 0.0],
                "oracle_alice_strategy_prob": 0.5,
                "exp3_alice_empirical_prob": 0.55,
                "no_attack_key_rate": 1.0,
                "worst_case_key_rate": 0.5,
            },
            {
                "graph_id": "g_parallel",
                "family": "disjoint_parallel",
                "route_id": "r_diag",
                "features": [2.0, 820.0, 1.025, 1.0, 0.0, 2.0, 0.0],
                "oracle_alice_strategy_prob": 0.0,
                "exp3_alice_empirical_prob": 0.0,
                "no_attack_key_rate": 0.8,
                "worst_case_key_rate": 0.4,
            },
        ],
        "trees": {
            "graph_oracle_retention": {
                "target": "oracle_retention",
                "feature_names": ["bottleneck_node_count", "node_disjoint_paths"],
                "r2_score": 1.0,
                "num_rows": 2,
                "num_topologies": 2,
                "rules_text": (
                    "IF bottleneck_node_count <= 0.500\n"
                    "  -> avg_oracle_retention = 0.500  (n=1)\n\n"
                    "IF bottleneck_node_count > 0.500\n"
                    "  -> avg_oracle_retention = 0.000  (n=1)"
                ),
            },
            "action_oracle_eve_strategy_prob": {
                "target": "oracle_eve_strategy_prob",
                "feature_names": [
                    "is_no_attack",
                    "is_edge_intercept_resend",
                    "is_memory_degradation",
                    "target_route_coverage",
                    "target_is_bottleneck",
                    "num_routes",
                    "node_disjoint_paths",
                ],
                "r2_score": 0.9,
                "num_rows": 2,
                "num_topologies": 2,
                "rules_text": (
                    "IF target_route_coverage <= 0.750\n"
                    "  -> avg_oracle_eve_strategy_prob = 0.500  (n=1)\n\n"
                    "IF target_route_coverage > 0.750\n"
                    "  -> avg_oracle_eve_strategy_prob = 1.000  (n=1)"
                ),
            },
            "action_expected_denial_under_oracle_alice": {
                "target": "action_expected_denial_under_oracle_alice",
                "feature_names": [
                    "is_no_attack",
                    "is_edge_intercept_resend",
                    "is_memory_degradation",
                    "target_route_coverage",
                    "target_is_bottleneck",
                    "num_routes",
                    "node_disjoint_paths",
                ],
                "r2_score": 0.8,
                "num_rows": 2,
                "num_topologies": 2,
                "rules_text": (
                    "IF target_route_coverage <= 0.750\n"
                    "  -> avg_action_expected_denial_under_oracle_alice = 0.500  (n=1)\n\n"
                    "IF target_route_coverage > 0.750\n"
                    "  -> avg_action_expected_denial_under_oracle_alice = 1.000  (n=1)"
                ),
            },
            "route_oracle_alice_strategy_prob": {
                "target": "oracle_alice_strategy_prob",
                "feature_names": [
                    "hop_count",
                    "length_m",
                    "length_over_shortest",
                    "internal_node_count",
                    "contains_bottleneck",
                    "node_disjoint_paths",
                    "mean_overlap_with_other_routes",
                ],
                "r2_score": 0.2,
                "num_rows": 2,
                "num_topologies": 2,
                "rules_text": (
                    "IF contains_bottleneck <= 0.500\n"
                    "  -> avg_oracle_alice_strategy_prob = 0.500  (n=1)\n\n"
                    "IF contains_bottleneck > 0.500\n"
                    "  -> avg_oracle_alice_strategy_prob = 1.000  (n=1)"
                ),
            },
            "route_exp3_alice_empirical_prob": {
                "target": "exp3_alice_empirical_prob",
                "feature_names": [
                    "hop_count",
                    "length_m",
                    "length_over_shortest",
                    "internal_node_count",
                    "contains_bottleneck",
                    "node_disjoint_paths",
                    "mean_overlap_with_other_routes",
                ],
                "r2_score": 0.2,
                "num_rows": 4,
                "num_topologies": 2,
                "rules_text": (
                    "IF contains_bottleneck <= 0.500\n"
                    "  -> avg_exp3_alice_empirical_prob = 0.275  (n=2)\n\n"
                    "IF contains_bottleneck > 0.500\n"
                    "  -> avg_exp3_alice_empirical_prob = 0.400  (n=2)"
                ),
            },
        },
    }
    oracle_summary = {
        "g_bottle": {
            "status": "absolute_eve_win",
            "retention": 0.0,
            "value": 0.0,
            "baseline_rate": 1.0,
        },
        "g_parallel": {
            "status": "meaningful_degradation",
            "retention": 0.5,
            "value": 0.5,
            "baseline_rate": 1.0,
        },
    }
    run_summary = {
        "backend": "sequence_repeater_e91",
        "security_monitor": "chsh",
        "trials_per_cell": 4,
        "online_turns": 10,
    }
    config = {"config": {"security_monitor": "chsh", "online_turns": 10}}
    _write_json(run_dir / "corpus.json", corpus)
    _write_json(run_dir / "dt" / "dt_payload.json", dt_payload)
    _write_json(run_dir / "oracle_summary.json", oracle_summary)
    _write_json(run_dir / "run_summary.json", run_summary)
    _write_json(run_dir / "config.json", config)


def _graph(graph_id, family, path):
    edges = [
        {
            "edge_id": f"e{index}",
            "u": u,
            "v": v,
            "length_m": 400.0,
            "eve_eligible": True,
        }
        for index, (u, v) in enumerate(zip(path, path[1:]))
    ]
    return {
        "graph_id": graph_id,
        "family": family,
        "alice": "a",
        "bob": "b",
        "features": {},
        "topology": {"edges": edges},
        "routes": [
            {
                "route_id": "r0",
                "path": path,
                "hop_count": len(path) - 1,
                "total_length_m": 400.0 * (len(path) - 1),
                "internal_nodes": path[1:-1],
            }
        ],
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
