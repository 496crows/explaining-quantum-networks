from __future__ import annotations

import json

import pytest

from sequence_game.experiments.exp3_sequence.route_verification import (
    RouteVerificationCase,
    RouteVerificationConfig,
    RouteVerificationResult,
    build_route_verification_cases,
    fixed_route_path,
    run_route_verification,
    summarize_route_verification,
)
from sequence_game.experiments.exp3_sequence.config import Exp3SequenceConfig


def _fake_result(
        case: RouteVerificationCase,
        config: RouteVerificationConfig,
) -> RouteVerificationResult:
    qualified = (
        case.stage == "hop_sanity"
        and case.edge_length_m <= 500.0
    ) or (
        case.stage == "frequency_benchmark"
        and case.memory_frequency_scale >= 2.0
    )
    delivered = config.min_delivered_pairs if qualified else 100
    return RouteVerificationResult(
        case=case,
        public_outcome="accepted" if qualified else "chsh_abort",
        accepted=qualified,
        qualified=qualified,
        delivered_count=delivered,
        qber=0.01 if qualified else 0.20,
        chsh_s=config.min_chsh_s + 0.1 if qualified else config.min_chsh_s - 0.1,
        chsh_adequately_sampled=qualified,
        wall_seconds=1.0,
        simulated_window_seconds=10.0,
        delivered_per_wall_second=float(delivered),
        delivered_per_simulated_second=float(delivered) / 10.0,
        timing={
            "start_time_ps": 1,
            "end_time_ps": 2,
            "stop_time_ps": 3,
        },
        active_models={
            "source_model_unused_by_repeater_path": True,
            "memory_frequency_hz": 76_000_000.0 * case.memory_frequency_scale,
            "memory_efficiency_override": 0.544,
        },
    )


def test_fixed_route_path_uses_requested_hop_count_and_lengths():
    path = fixed_route_path(4, 125.0)
    assert path.nodes == ("alice", "r1", "r2", "r3", "bob")
    assert path.edge_lengths_m == pytest.approx((125.0, 125.0, 125.0, 125.0))


def test_route_verification_case_grid_covers_lengths_and_frequency_diagnostic():
    config = RouteVerificationConfig(
        max_hops=3,
        hop_counts=(1, 2, 3),
        edge_lengths_m=(1000.0, 500.0),
        frequency_scales=(1.0, 2.0),
    )
    cases = build_route_verification_cases(config)
    length_cases = [case for case in cases if case.stage == "hop_sanity"]
    frequency_cases = [case for case in cases if case.stage == "frequency_benchmark"]

    assert len(length_cases) == 6
    assert {(case.hops, case.edge_length_m) for case in length_cases} == {
        (1, 1000.0), (1, 500.0),
        (2, 1000.0), (2, 500.0),
        (3, 1000.0), (3, 500.0),
    }
    assert {(case.hops, case.memory_frequency_scale) for case in frequency_cases} == {
        (1, 2.0), (2, 2.0), (3, 2.0),
    }


def test_default_route_verification_checks_one_case_per_hop_class():
    config = RouteVerificationConfig()
    cases = build_route_verification_cases(config)

    assert config.sequence_memory_fidelity_override == pytest.approx(1.0)
    assert len(cases) == 7
    assert [case.hops for case in cases] == list(range(1, 8))
    assert {case.stage for case in cases} == {"hop_sanity"}
    assert {case.edge_length_m for case in cases} == {500.0}


def test_route_verification_from_exp3_config_uses_sqlite_corpus_depth():
    config = RouteVerificationConfig.from_exp3_config(Exp3SequenceConfig())

    assert config.max_hops == 7
    assert config.hop_counts == tuple(range(1, 8))
    assert config.sequence_memory_fidelity_override == pytest.approx(1.0)


def test_route_verification_summary_reports_max_qualified_edge_by_hop():
    config = RouteVerificationConfig(
        max_hops=2,
        hop_counts=(1, 2),
        edge_lengths_m=(1000.0, 500.0),
        frequency_scales=(1.0,),
    )
    results = [_fake_result(case, config) for case in build_route_verification_cases(config)]
    summary = summarize_route_verification(results)

    assert summary["length_thresholds_by_hop"]["1"]["max_passing_edge_length_m"] == 500.0
    assert summary["length_thresholds_by_hop"]["2"]["max_passing_edge_length_m"] == 500.0
    assert summary["hop_sanity_by_hop"] == summary["length_thresholds_by_hop"]
    assert summary["qualified_count"] == 2


def test_route_verification_writes_json_without_running_sequence(tmp_path):
    config = RouteVerificationConfig(
        out_dir=tmp_path / "route_verify",
        max_hops=2,
        hop_counts=(1, 2),
        workers=1,
        edge_lengths_m=(1000.0, 500.0),
        frequency_scales=(1.0, 2.0),
    )
    payload = run_route_verification(config, progress=False, trial_runner=_fake_result)
    output = tmp_path / "route_verify" / "route_verification.json"

    assert output.exists()
    saved = json.loads(output.read_text())
    assert saved["scope"]["route_verification_only"] is True
    assert saved["scope"]["source_model_active_in_repeater_path"] is False
    assert saved["summary"] == payload["summary"]
