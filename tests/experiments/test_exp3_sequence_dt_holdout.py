from sequence_game.experiments.exp3_sequence.dt_holdout import (
    evaluate_dt_holdout_payload,
    markdown_table,
    strong_generalization_drops,
)


def test_evaluate_dt_holdout_payload_reports_all_oracle_targets():
    payload = {
        "graph_rows": [
            _row("g0", [0.0, 0.0], "oracle_retention", 0.0),
            _row("g1", [1.0, 1.0], "oracle_retention", 0.5),
            _row("g2", [2.0, 2.0], "oracle_retention", 1.0),
        ],
        "action_rows": [
            _row("g0", [0.0, 0.0], "oracle_eve_strategy_prob", 0.0),
            _row("g0", [0.0, 1.0], "oracle_eve_strategy_prob", 0.2),
            _row("g1", [1.0, 0.0], "oracle_eve_strategy_prob", 0.4),
            _row("g1", [1.0, 1.0], "oracle_eve_strategy_prob", 0.6),
            _row("g2", [2.0, 0.0], "oracle_eve_strategy_prob", 0.8),
            _row("g2", [2.0, 1.0], "oracle_eve_strategy_prob", 1.0),
            _row("g0", [0.0, 0.0], "action_expected_denial_under_oracle_alice", 1.0),
            _row("g0", [0.0, 1.0], "action_expected_denial_under_oracle_alice", 0.8),
            _row("g1", [1.0, 0.0], "action_expected_denial_under_oracle_alice", 0.6),
            _row("g1", [1.0, 1.0], "action_expected_denial_under_oracle_alice", 0.4),
            _row("g2", [2.0, 0.0], "action_expected_denial_under_oracle_alice", 0.2),
            _row("g2", [2.0, 1.0], "action_expected_denial_under_oracle_alice", 0.0),
        ],
        "route_rows": [
            _row("g0", [0.0, 0.0], "oracle_alice_strategy_prob", 0.1),
            _row("g0", [0.0, 1.0], "oracle_alice_strategy_prob", 0.9),
            _row("g1", [1.0, 0.0], "oracle_alice_strategy_prob", 0.2),
            _row("g1", [1.0, 1.0], "oracle_alice_strategy_prob", 0.8),
            _row("g2", [2.0, 0.0], "oracle_alice_strategy_prob", 0.3),
            _row("g2", [2.0, 1.0], "oracle_alice_strategy_prob", 0.7),
        ],
    }

    results = evaluate_dt_holdout_payload(payload, max_depth=1)

    assert [result.target for result in results] == [
        "Oracle graph retention",
        "Oracle Eve action probability",
        "Expected denial vs. oracle Alice",
        "Oracle Alice route probability",
    ]
    assert all(result.topologies == 3 for result in results)
    assert all(result.rows > 0 for result in results)
    assert "Topology-held-out R2" in markdown_table(results)
    assert isinstance(strong_generalization_drops(results), list)


def _row(graph_id: str, features: list[float], target: str, value: float) -> dict:
    return {
        "graph_id": graph_id,
        "features": features,
        target: value,
    }
