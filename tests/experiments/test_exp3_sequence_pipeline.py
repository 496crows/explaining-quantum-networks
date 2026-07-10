from __future__ import annotations

import json

import numpy as np
import pytest

from sequence_game.experiments.exp3_sequence.backend import (
    ActionSpec,
    TurnResult,
    build_actions,
)
from sequence_game.corpus.e91_runtime_game import _default_models
from sequence_game.experiments.exp3_sequence.attack_cache import (
    attack_cache_hop_statistics,
    attack_cache_summary,
    build_attack_tasks,
    initialize_attack_cache,
    insert_attack_sample,
    load_cached_attack_samples,
)
from sequence_game.experiments.exp3_sequence.config import (
    CORPUS_SQLITE_PATH,
    Exp3SequenceConfig,
)
from sequence_game.experiments.exp3_sequence.learners import (
    CautiousGreedyAlicePolicy,
    make_policy,
)
from sequence_game.experiments.exp3_sequence.baseline_cache import (
    BaselineCacheTask,
    baseline_cache_hop_statistics,
    initialize_baseline_cache,
    insert_baseline_sample,
    load_cached_baseline_results,
    normalize_case_payload,
    route_physics_cache_key,
    route_physics_cache_key_from_vector,
    standard_route_length_vector,
)
from sequence_game.experiments.exp3_sequence.corpus import (
    GraphCase,
    build_graph_cases,
    load_graph_cases_from_sqlite,
    read_corpus_sqlite_metadata,
    write_graph_cases_sqlite,
)
from sequence_game.experiments.exp3_sequence.dt_rows import action_strategy_rows
from sequence_game.experiments.exp3_sequence.models import (
    apply_sequence_memory_fidelity_override,
)
from sequence_game.experiments.exp3_sequence.online import payoff_turn_result
from sequence_game.experiments.exp3_sequence.oracle import solve_oracle
from sequence_game.experiments.exp3_sequence.payoff import PayoffEstimate, summarize_cell
from sequence_game.experiments.exp3_sequence.runner import run_pipeline


def test_corpus_contains_disjoint_and_bottleneck_cases():
    cases = build_graph_cases(
        max_graphs=10,
        max_routes=64,
        max_route_hops=8,
        base_seed=123,
    )
    families = {case.family for case in cases}
    assert "disjoint_parallel" in families
    assert "single_bottleneck" in families
    assert all(case.routes for case in cases)
    assert max(case.features["num_routes"] for case in cases) >= 4


def test_sqlite_corpus_round_trips_graph_cases(tmp_path):
    source_cases = build_graph_cases(
        max_graphs=3,
        max_routes=8,
        max_route_hops=7,
        base_seed=123,
    )
    db_path = tmp_path / "corpus.sqlite"
    write_graph_cases_sqlite(
        db_path,
        source_cases,
        metadata={"graph_count": len(source_cases), "max_route_hops": 7},
    )

    loaded = load_graph_cases_from_sqlite(db_path)
    metadata = read_corpus_sqlite_metadata(db_path)

    assert [case.graph_id for case in loaded] == [
        case.graph_id for case in source_cases
    ]
    assert metadata["graph_count"] == 3
    assert metadata["max_route_hops"] == 7


def test_default_sqlite_corpus_reaches_32_route_scale():
    cases = load_graph_cases_from_sqlite(CORPUS_SQLITE_PATH)
    metadata = read_corpus_sqlite_metadata(CORPUS_SQLITE_PATH)

    assert len(cases) == 50
    assert metadata["graph_count"] == 50
    assert metadata["max_route_hops"] == 7
    assert metadata["max_routes_per_graph"] == 32
    assert max(case.features["num_routes"] for case in cases) == 32
    assert max(case.features["node_disjoint_paths"] for case in cases) == 32
    assert max(case.features["longest_hops"] for case in cases) == 7
    assert {case.family for case in cases} >= {
        "deep_parallel",
        "deep_bottleneck",
    }
    route_lengths_km = [
        route["total_length_m"] / 1000.0
        for case in cases
        for route in case.routes
    ]
    assert 1.3 < np.mean(route_lengths_km) < 2.4
    assert max(route_lengths_km) < 3.8


def test_standard_route_length_vectors_are_canonical_by_hop_count():
    assert standard_route_length_vector(1) == pytest.approx((400.0,))
    assert standard_route_length_vector(4) == pytest.approx(
        (400.0, 425.0, 450.0, 475.0))


def test_normalize_case_payload_adds_route_edge_length_vectors():
    case = load_graph_cases_from_sqlite(CORPUS_SQLITE_PATH, limit=1)[0]
    payload = normalize_case_payload(case.to_dict())

    for route in payload["routes"]:
        expected = standard_route_length_vector(int(route["hop_count"]))
        assert route["edge_lengths_m"] == pytest.approx(expected)
        assert route["total_length_m"] == pytest.approx(sum(expected))
        assert route["length_profile_id"] == f"standard_h{route['hop_count']}"
    assert payload["topology"]["metadata"]["params"][
        "route_edge_lengths_are_canonical_for_sequence_runtime"
    ] is True


def test_baseline_cache_round_trips_turn_results(tmp_path):
    db_path = tmp_path / "baselines.sqlite"
    initialize_baseline_cache(db_path, metadata={"test": True})
    vector = standard_route_length_vector(2)
    task = BaselineCacheTask(
        hop_count=2,
        sample_index=0,
        length_vector_m=vector,
        seed=123,
    )
    result = TurnResult(
        public_outcome="accepted",
        alice_reward=1.0,
        eve_hit_reward=0.0,
        active_route_attacked=False,
        accepted=True,
        qber=0.01,
        chsh_s=2.7,
        chsh_adequately_sampled=True,
        delivered_count=350,
        sifted_count=350,
        fidelity=1.0,
        runtime_engine="fake_sequence_repeater_chsh_trial",
        runtime_attack_applied={"kind": "no_attack"},
        sequence_timing={"start_time_ps": 1},
    )

    insert_baseline_sample(
        db_path, task=task, result=result, wall_seconds=0.5)
    cached = load_cached_baseline_results(
        db_path,
        cache_key=route_physics_cache_key_from_vector(vector),
        sample_count=1,
        seed=123,
    )

    assert cached is not None
    assert len(cached) == 1
    assert cached[0].public_outcome == "accepted"
    assert cached[0].chsh_s == pytest.approx(2.7)
    rows = baseline_cache_hop_statistics(db_path)
    assert len(rows) == 1
    assert rows[0]["hop_count"] == 2
    assert rows[0]["samples"] == 1
    assert rows[0]["mean_abs_chsh_s"] == pytest.approx(2.7)
    assert rows[0]["mean_qber"] == pytest.approx(0.01)


def test_attack_tasks_are_linear_and_target_balanced():
    tasks = build_attack_tasks(max_hops=3, samples_per_hop=8, seed=77)

    assert len(tasks) == 40
    assert sum(1 for task in tasks if task.attack_kind == "edge_intercept_resend") == 24
    assert sum(1 for task in tasks if task.attack_kind == "memory_degradation") == 16
    assert not any(
        task.attack_kind == "memory_degradation" and task.hop_count == 1
        for task in tasks
    )

    for attack_kind, hop_count in {
        (task.attack_kind, task.hop_count) for task in tasks
    }:
        group = [
            task for task in tasks
            if task.attack_kind == attack_kind and task.hop_count == hop_count
        ]
        counts = {
            index: sum(1 for task in group if task.target_index == index)
            for index in range(group[0].target_count)
        }
        assert min(counts.values()) >= 1
        assert max(counts.values()) - min(counts.values()) <= 1


def test_attack_cache_round_trips_target_metadata_and_signals(tmp_path):
    db_path = tmp_path / "attack_baselines.sqlite"
    initialize_attack_cache(db_path, metadata={"test": True})
    task = [
        candidate for candidate in build_attack_tasks(
            max_hops=3,
            samples_per_hop=4,
            seed=99,
            attack_kinds=("edge_intercept_resend",),
        )
        if candidate.hop_count == 3
    ][0]
    result = TurnResult(
        public_outcome="chsh_abort",
        alice_reward=0.0,
        eve_hit_reward=1.0,
        active_route_attacked=True,
        accepted=False,
        qber=0.23,
        chsh_s=1.91,
        chsh_adequately_sampled=True,
        delivered_count=350,
        sifted_count=120,
        fidelity=0.88,
        runtime_engine="fake_sequence_repeater_chsh_trial",
        runtime_attack_applied=task.action.to_dict(),
        sequence_timing={"start_time_ps": 1},
    )

    insert_attack_sample(
        db_path,
        task=task,
        result=result,
        wall_seconds=0.25,
        eve_information_status="hit_and_location_only",
    )
    samples = load_cached_attack_samples(
        db_path,
        cache_key=task.cache_key,
        sample_count=1,
        seed=123,
    )
    summary = attack_cache_summary(db_path)

    assert samples is not None
    assert len(samples) == 1
    sample = samples[0]
    assert sample.attack_kind == "edge_intercept_resend"
    assert sample.target_kind == "edge"
    assert sample.target_id == task.target_id
    assert sample.result.chsh_s == pytest.approx(1.91)
    assert sample.result.qber == pytest.approx(0.23)
    assert sample.result.eve_hit_reward == pytest.approx(1.0)
    assert summary["sample_count"] == 1
    assert summary["by_attack_kind"]["edge_intercept_resend"]["3"][
        "mean_qber"
    ] == pytest.approx(0.23)


def test_attack_cache_hop_statistics_preserves_signed_and_abs_chsh(tmp_path):
    db_path = tmp_path / "attack_baselines.sqlite"
    initialize_attack_cache(db_path, metadata={"test": True})
    tasks = [
        task for task in build_attack_tasks(
            max_hops=1,
            samples_per_hop=2,
            seed=123,
            attack_kinds=("edge_intercept_resend",),
        )
    ]
    for task, chsh_s in zip(tasks, (-0.5, -1.5)):
        insert_attack_sample(
            db_path,
            task=task,
            result=TurnResult(
                public_outcome="chsh_abort",
                alice_reward=0.0,
                eve_hit_reward=1.0,
                active_route_attacked=True,
                accepted=False,
                qber=0.50,
                chsh_s=chsh_s,
                chsh_adequately_sampled=True,
                delivered_count=350,
                sifted_count=120,
                fidelity=0.88,
                runtime_engine="fake_sequence_repeater_chsh_trial",
                runtime_attack_applied=task.action.to_dict(),
                sequence_timing={"start_time_ps": 1},
            ),
            wall_seconds=0.25,
            eve_information_status="hit_and_location_only",
        )

    rows = attack_cache_hop_statistics(db_path)

    assert len(rows) == 1
    assert rows[0]["attack_kind"] == "edge_intercept_resend"
    assert rows[0]["hop_count"] == 1
    assert rows[0]["samples"] == 2
    assert rows[0]["mean_chsh_s"] == pytest.approx(-1.0)
    assert rows[0]["mean_abs_chsh_s"] == pytest.approx(1.0)
    assert rows[0]["mean_qber"] == pytest.approx(0.50)


def test_health_check_uses_cached_baselines_without_sequence(monkeypatch, tmp_path):
    import sequence_game.experiments.exp3_sequence.health as health_mod

    class RaisingEvaluator:
        def __init__(self, ir_dict, routes, config):
            raise AssertionError("health should use cached baseline rows")

    monkeypatch.setattr(health_mod, "SequenceRouteEvaluator", RaisingEvaluator)

    case = load_graph_cases_from_sqlite(CORPUS_SQLITE_PATH, limit=1)[0]
    route = case.routes[0]
    vector = tuple(float(value) for value in route["edge_lengths_m"])
    db_path = tmp_path / "baselines.sqlite"
    initialize_baseline_cache(db_path, metadata={"test": True})
    for sample_index in range(2):
        task = BaselineCacheTask(
            hop_count=int(route["hop_count"]),
            sample_index=sample_index,
            length_vector_m=vector,
            seed=1_000 + sample_index,
        )
        result = TurnResult(
            public_outcome="accepted",
            alice_reward=1.0,
            eve_hit_reward=0.0,
            active_route_attacked=False,
            accepted=True,
            qber=0.0,
            chsh_s=2.75,
            chsh_adequately_sampled=True,
            delivered_count=350,
            sifted_count=350,
            fidelity=1.0,
            runtime_engine="cached_sequence_repeater_chsh_trial",
            runtime_attack_applied={"kind": "no_attack"},
            sequence_timing={"cached": True},
        )
        insert_baseline_sample(
            db_path, task=task, result=result, wall_seconds=0.1)

    cached = load_cached_baseline_results(
        db_path,
        cache_key=route_physics_cache_key(route),
        sample_count=2,
        seed=99,
    )
    assert cached is not None

    config = Exp3SequenceConfig(
        baseline_cache_db_path=db_path,
        baseline_health_routes_per_graph=1,
        baseline_health_trials_per_route=2,
    )
    health = health_mod.run_graph_health_check(case, config, seed_base=99)

    assert health.healthy is True
    assert health.trial_count == 2
    assert health.accepted_count == 2
    assert health.route_results[0].result_source == "baseline_cache"


def test_default_exp3_sequence_uses_ideal_memory_fidelity_override():
    _source, _detector, _fiber, memory = _default_models()
    assert memory.parameters["fidelity"] == pytest.approx(0.933)

    active = apply_sequence_memory_fidelity_override(memory, Exp3SequenceConfig())

    assert active.parameters["fidelity"] == pytest.approx(1.0)
    assert active.parameters["sequence_runtime_original_fidelity"] == pytest.approx(0.933)
    assert active.parameters["sequence_runtime_fidelity_override"] == pytest.approx(1.0)


def test_default_action_surface_is_edge_intercept_and_memory_degradation_only():
    cases = load_graph_cases_from_sqlite(CORPUS_SQLITE_PATH, limit=1)
    actions = build_actions(cases[0].routes, Exp3SequenceConfig().action_kinds)
    attack_types = {action.attack_type for action in actions}

    assert attack_types == {
        "no_attack",
        "edge_intercept_resend",
        "memory_degradation",
    }
    assert {action.kind for action in actions} == {"none", "edge", "node"}


def test_default_conditions_include_cautious_greedy_control():
    by_key = {
        condition.key: (condition.alice, condition.eve)
        for condition in Exp3SequenceConfig().conditions
    }

    assert by_key["exp3_vs_exp3"] == ("exp3_bandit", "exp3_bandit")
    assert by_key["exp3_eve_vs_cautious_greedy_alice"] == (
        "cautious_greedy",
        "exp3_bandit",
    )


def test_exp3_config_uses_separate_alice_and_eve_gamma():
    config = Exp3SequenceConfig(alice_exp3_gamma=0.03, eve_exp3_gamma=0.11)
    payload = config.to_dict()

    assert config.alice_exp3_gamma == pytest.approx(0.03)
    assert config.eve_exp3_gamma == pytest.approx(0.11)
    assert payload["exp3_gamma"] == pytest.approx({
        "alice": 0.03,
        "eve": 0.11,
    })
    assert payload["exp3_schedule"]["mode"] == "constant"
    assert payload["alice_acceptance_rule"] == "chsh_only"


def test_exp3_config_accepts_anytime_schedule_parameters():
    config = Exp3SequenceConfig(
        exp3_schedule_mode="anytime",
        alice_exp3_eta_c=0.9,
        eve_exp3_eta_c=1.1,
        alice_exp3_t0=100.0,
        eve_exp3_t0=200.0,
        alice_exp3_gamma_max=0.15,
        eve_exp3_gamma_max=0.18,
    )
    payload = config.to_dict()["exp3_schedule"]

    assert payload["mode"] == "anytime"
    assert payload["alice"]["eta_c"] == pytest.approx(0.9)
    assert payload["eve"]["eta_c"] == pytest.approx(1.1)
    assert payload["alice"]["t0"] == pytest.approx(100.0)
    assert payload["eve"]["t0"] == pytest.approx(200.0)
    assert payload["alice"]["gamma_max"] == pytest.approx(0.15)
    assert payload["eve"]["gamma_max"] == pytest.approx(0.18)


def test_make_policy_receives_side_specific_gamma():
    alice = make_policy(
        "exp3_bandit",
        3,
        oracle_strategy=np.asarray([1 / 3, 1 / 3, 1 / 3]),
        gamma=0.03,
        rng=np.random.default_rng(1),
    )
    eve = make_policy(
        "exp3_bandit",
        5,
        oracle_strategy=np.asarray([0.2] * 5),
        gamma=0.11,
        rng=np.random.default_rng(2),
    )

    assert alice.learner.gamma == pytest.approx(0.03)
    assert eve.learner.gamma == pytest.approx(0.11)


def test_make_policy_receives_anytime_schedule_parameters():
    policy = make_policy(
        "exp3_bandit",
        4,
        oracle_strategy=np.asarray([0.25] * 4),
        gamma=0.07,
        rng=np.random.default_rng(1),
        schedule_mode="anytime",
        eta_c=0.8,
        t0=50.0,
        gamma_max=0.13,
    )

    assert policy.learner.schedule_mode == "anytime"
    assert policy.learner.eta_c == pytest.approx(0.8)
    assert policy.learner.t0 == pytest.approx(50.0)
    assert policy.learner.gamma_max == pytest.approx(0.13)


def test_chsh_only_acceptance_ignores_qber_veto_but_keeps_qber_diagnostics():
    qber_abort = TurnResult(
        public_outcome="qber_abort",
        alice_reward=0.0,
        eve_hit_reward=0.0,
        active_route_attacked=False,
        accepted=False,
        qber=0.42,
        chsh_s=2.35,
        chsh_adequately_sampled=True,
        delivered_count=350,
        sifted_count=350,
        fidelity=0.98,
        runtime_engine="cached_fake_sequence_repeater_chsh_trial",
        runtime_attack_applied={"kind": "no_attack"},
        sequence_timing={"route_total_length_m": 400.0},
    )

    chsh_only = summarize_cell(
        route_id="r0",
        action_id="no_attack",
        results=[qber_abort],
        seed_start=1,
        seed_end=1,
        ci_half_width_warn=1.0,
        simulated_action_id="no_attack",
        config=Exp3SequenceConfig(alice_acceptance_rule="chsh_only"),
    )
    strict = summarize_cell(
        route_id="r0",
        action_id="no_attack",
        results=[qber_abort],
        seed_start=1,
        seed_end=1,
        ci_half_width_warn=1.0,
        simulated_action_id="no_attack",
        config=Exp3SequenceConfig(alice_acceptance_rule="chsh_and_qber"),
    )

    assert chsh_only.accepted_count == 1
    assert chsh_only.accepted_rate == pytest.approx(1.0)
    assert chsh_only.qber_abort_count == 1
    assert chsh_only.qber_count == 1
    assert chsh_only.mean_qber == pytest.approx(0.42)
    assert chsh_only.sequence_timing["qber_is_diagnostic_not_hard_veto"] is True
    assert chsh_only.sequence_timing["no_attack_false_signal_count"] == 1
    assert strict.accepted_count == 0
    assert strict.sequence_timing["qber_is_diagnostic_not_hard_veto"] is False


def test_cautious_greedy_alice_avoids_public_denials():
    policy = CautiousGreedyAlicePolicy(
        np.asarray([0.8, 1.0, 0.9]),
        avoidance_horizon=1,
    )

    assert policy.sample() == 1
    policy.update(0.0)
    policy.observe_public_outcome("qber_abort")
    assert policy.sample() == 2
    assert policy.last_fallback is False
    policy.update(1.0)
    policy.observe_public_outcome("accepted")
    assert policy.sample() == 1
    assert policy.empirical_strategy() == pytest.approx([0.0, 2 / 3, 1 / 3])


def test_cautious_greedy_alice_reports_fallback_when_all_routes_avoided():
    policy = CautiousGreedyAlicePolicy(
        np.asarray([1.0, 0.9]),
        avoidance_horizon=2,
    )

    assert policy.sample() == 0
    policy.observe_public_outcome("chsh_abort")
    assert policy.sample() == 1
    policy.observe_public_outcome("delivery_failure")
    assert policy.sample() == 0
    assert policy.last_fallback is True


def test_online_turn_samples_cached_clean_experiment_not_matrix_mean(tmp_path):
    case = load_graph_cases_from_sqlite(CORPUS_SQLITE_PATH, limit=1)[0]
    route = case.routes[0]
    db_path = tmp_path / "baselines.sqlite"
    initialize_baseline_cache(db_path, metadata={"test": True})
    task = BaselineCacheTask(
        hop_count=int(route["hop_count"]),
        sample_index=0,
        length_vector_m=tuple(float(value) for value in route["edge_lengths_m"]),
        seed=123,
    )
    insert_baseline_sample(
        db_path,
        task=task,
        result=TurnResult(
            public_outcome="delivery_failure",
            alice_reward=0.0,
            eve_hit_reward=0.0,
            active_route_attacked=False,
            accepted=False,
            qber=None,
            chsh_s=None,
            chsh_adequately_sampled=False,
            delivered_count=0,
            sifted_count=0,
            fidelity=None,
            runtime_engine="cached_fake_sequence_repeater_chsh_trial",
            runtime_attack_applied={"kind": "no_attack"},
            sequence_timing={"cached_turn": True},
        ),
        wall_seconds=0.01,
    )
    actions = [ActionSpec("no_attack", "none", "", "no_attack")]
    cell = summarize_cell(
        route_id=str(route["route_id"]),
        action_id="no_attack",
        results=[
            TurnResult(
                public_outcome="accepted",
                alice_reward=1.0,
                eve_hit_reward=0.0,
                active_route_attacked=False,
                accepted=True,
                qber=0.0,
                chsh_s=2.82,
                chsh_adequately_sampled=True,
                delivered_count=350,
                sifted_count=350,
                fidelity=1.0,
                runtime_engine="averaged_cell",
                runtime_attack_applied={"kind": "no_attack"},
                sequence_timing={"cell_mean": True},
            )
        ],
        seed_start=1,
        seed_end=1,
        ci_half_width_warn=1.0,
    )
    payoff = PayoffEstimate(
        graph_id=case.graph_id,
        route_ids=(str(route["route_id"]),),
        action_ids=("no_attack",),
        payoff=np.asarray([[1.0]]),
        cells=(cell,),
        seed_start=1,
        seed_end=1,
    )
    config = Exp3SequenceConfig(
        baseline_cache_db_path=db_path,
        attack_cache_db_path=tmp_path / "missing_attacks.sqlite",
    )

    result = payoff_turn_result(
        GraphCase(
            graph_id=case.graph_id,
            family=case.family,
            alice=case.alice,
            bob=case.bob,
            ir_dict=case.ir_dict,
            assignment=case.assignment,
            routes=[route],
            route_selection=case.route_selection,
            features=case.features,
        ),
        actions,
        payoff,
        config,
        0,
        0,
        seed=999,
    )

    assert result.public_outcome == "delivery_failure"
    assert result.alice_reward == pytest.approx(0.0)
    assert result.sequence_timing["online_model"] == "cached_sequence_game_turn_sample"
    assert result.sequence_timing["online_sample_source"] == "clean_route_profile_cache"


def test_oracle_solves_disjoint_route_matrix():
    payoff = PayoffEstimate(
        graph_id="disjoint_3",
        route_ids=("r0", "r1", "r2"),
        action_ids=("a0", "a1", "a2"),
        payoff=1.0 - np.eye(3),
        cells=(),
        seed_start=1,
        seed_end=1,
    )
    oracle = solve_oracle(payoff)
    assert oracle.value == pytest.approx(2 / 3, abs=1e-6)
    assert oracle.retention == pytest.approx(2 / 3, abs=1e-6)
    assert oracle.alice_strategy == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)
    assert oracle.eve_strategy == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)


def test_oracle_status_distinguishes_minor_degradation_from_eve_win():
    payoff = PayoffEstimate(
        graph_id="minor_loss",
        route_ids=("r0",),
        action_ids=("no_attack", "weak_attack"),
        payoff=np.asarray([[0.750, 0.703]]),
        cells=(),
        seed_start=1,
        seed_end=1,
    )

    oracle = solve_oracle(payoff)

    assert oracle.retention == pytest.approx(0.703 / 0.750, abs=1e-6)
    assert oracle.to_dict()["retention_loss"] == pytest.approx(
        1.0 - 0.703 / 0.750,
        abs=1e-6,
    )
    assert oracle.status == "minor_degradation"


def test_oracle_status_marks_meaningful_and_absolute_degradation():
    meaningful = solve_oracle(PayoffEstimate(
        graph_id="meaningful_loss",
        route_ids=("r0",),
        action_ids=("no_attack", "strong_attack"),
        payoff=np.asarray([[0.750, 0.400]]),
        cells=(),
        seed_start=1,
        seed_end=1,
    ))
    absolute = solve_oracle(PayoffEstimate(
        graph_id="absolute_loss",
        route_ids=("r0",),
        action_ids=("no_attack", "full_denial"),
        payoff=np.asarray([[0.750, 0.0]]),
        cells=(),
        seed_start=1,
        seed_end=1,
    ))

    assert meaningful.status == "meaningful_degradation"
    assert absolute.status == "absolute_eve_win"


def test_action_dt_rows_include_denial_under_oracle_alice():
    cases = build_graph_cases(
        max_graphs=1,
        max_routes=2,
        max_route_hops=8,
        base_seed=123,
    )
    case = cases[0]
    actions = [
        ActionSpec("no_attack", "none", "", "no_attack"),
        ActionSpec(
            "deny_r0",
            "node",
            case.routes[0]["internal_nodes"][0],
            "memory_degradation",
        ),
        ActionSpec(
            "deny_r1",
            "node",
            case.routes[1]["internal_nodes"][0],
            "memory_degradation",
        ),
    ]
    payoff = PayoffEstimate(
        graph_id=case.graph_id,
        route_ids=tuple(str(route["route_id"]) for route in case.routes),
        action_ids=tuple(action.action_id for action in actions),
        payoff=np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        cells=(),
        seed_start=1,
        seed_end=1,
    )
    oracle = solve_oracle(payoff)
    rows = action_strategy_rows(
        [case],
        {case.graph_id: payoff},
        {case.graph_id: actions},
        {case.graph_id: oracle},
        {},
    )
    denial_by_action = {
        row["action_id"]: row["action_expected_denial_under_oracle_alice"]
        for row in rows
    }
    assert denial_by_action["no_attack"] == pytest.approx(0.0)
    assert denial_by_action["deny_r0"] == pytest.approx(0.5, abs=1e-6)
    assert denial_by_action["deny_r1"] == pytest.approx(0.5, abs=1e-6)


class FakeEvaluator:
    def __init__(self, ir_dict, routes, config):
        self.routes = routes

    def evaluate(self, route_index: int, action: ActionSpec, *, seed: int, trial_id: str):
        assert seed >= 0
        route = self.routes[route_index]
        if action.kind == "edge":
            path = [str(node) for node in route.get("path", [])]
            edges = {
                f"{min(u, v)}-{max(u, v)}"
                for u, v in zip(path, path[1:])
            }
            edges.update(str(edge_id) for edge_id in route.get("edge_ids", ()))
            active = action.target in edges
        else:
            active = action.target in set(route.get("internal_nodes") or [])
        denied = active and action.attack_type != "no_attack"
        return TurnResult(
            public_outcome="delivery_failure" if denied else "accepted",
            alice_reward=0.0 if denied else 1.0,
            eve_hit_reward=1.0 if denied else 0.0,
            active_route_attacked=active,
            accepted=not denied,
            qber=0.0 if not denied else None,
            chsh_s=2.82 if not denied else None,
            chsh_adequately_sampled=not denied,
            delivered_count=350 if not denied else 0,
            sifted_count=350 if not denied else 0,
            fidelity=0.99 if not denied else None,
            runtime_engine="fake_sequence_repeater_chsh_trial",
            runtime_attack_applied={
                "kind": action.attack_type,
                "target_node": action.target,
            },
            sequence_timing={
                "start_time_ps": 1,
                "end_time_ps": 2,
                "stop_time_ps": 3,
                "route_total_length_m": route["total_length_m"],
            },
        )


class FailingHealthEvaluator(FakeEvaluator):
    def evaluate(self, route_index: int, action: ActionSpec, *, seed: int, trial_id: str):
        route = self.routes[route_index]
        return TurnResult(
            public_outcome="chsh_abort",
            alice_reward=0.0,
            eve_hit_reward=0.0,
            active_route_attacked=False,
            accepted=False,
            qber=None,
            chsh_s=1.8,
            chsh_adequately_sampled=True,
            delivered_count=1,
            sifted_count=1,
            fidelity=0.99,
            runtime_engine="fake_sequence_repeater_chsh_trial",
            runtime_attack_applied={"kind": "no_attack", "target_node": ""},
            sequence_timing={
                "start_time_ps": 1,
                "end_time_ps": 2,
                "stop_time_ps": 3,
                "route_total_length_m": route["total_length_m"],
            },
        )


def _write_fake_attack_cache(db_path, *, samples_per_hop: int = 4) -> None:
    initialize_attack_cache(db_path, metadata={"test": True})
    for task in build_attack_tasks(samples_per_hop=samples_per_hop):
        edge_attack = task.attack_kind == "edge_intercept_resend"
        result = TurnResult(
            public_outcome="chsh_abort" if edge_attack else "qber_abort",
            alice_reward=0.0,
            eve_hit_reward=1.0,
            active_route_attacked=True,
            accepted=False,
            qber=0.55 if edge_attack else 0.30,
            chsh_s=-0.5 if edge_attack else 1.2,
            chsh_adequately_sampled=True,
            delivered_count=350,
            sifted_count=350,
            fidelity=0.70,
            runtime_engine="fake_sequence_repeater_chsh_trial",
            runtime_attack_applied=task.action.to_dict(),
            sequence_timing={
                "start_time_ps": 1,
                "end_time_ps": 2,
                "stop_time_ps": 3,
                "route_total_length_m": sum(task.length_vector_m),
                "attack_kind": task.attack_kind,
            },
        )
        insert_attack_sample(
            db_path,
            task=task,
            result=result,
            wall_seconds=0.01,
            eve_information_status="test_hit_location_only",
        )


def _write_fake_baseline_cache(db_path, *, samples_per_hop: int = 4) -> None:
    initialize_baseline_cache(db_path, metadata={"test": True})
    for hop_count in range(1, 8):
        vector = standard_route_length_vector(hop_count)
        for sample_index in range(samples_per_hop):
            task = BaselineCacheTask(
                hop_count=hop_count,
                sample_index=sample_index,
                length_vector_m=vector,
                seed=10_000 + hop_count * 100 + sample_index,
            )
            result = TurnResult(
                public_outcome="accepted",
                alice_reward=1.0,
                eve_hit_reward=0.0,
                active_route_attacked=False,
                accepted=True,
                qber=0.0,
                chsh_s=2.82,
                chsh_adequately_sampled=True,
                delivered_count=350,
                sifted_count=350,
                fidelity=1.0,
                runtime_engine="cached_fake_sequence_repeater_chsh_trial",
                runtime_attack_applied={"kind": "no_attack"},
                sequence_timing={
                    "start_time_ps": 1,
                    "end_time_ps": 2,
                    "stop_time_ps": 3,
                    "route_total_length_m": sum(vector),
                },
            )
            insert_baseline_sample(
                db_path,
                task=task,
                result=result,
                wall_seconds=0.01,
            )


def test_pipeline_writes_runnable_outputs(monkeypatch, tmp_path):
    import sequence_game.experiments.exp3_sequence.health as health_mod
    import sequence_game.experiments.exp3_sequence.payoff as payoff_mod

    monkeypatch.setattr(health_mod, "SequenceRouteEvaluator", FakeEvaluator)
    monkeypatch.setattr(payoff_mod, "SequenceRouteEvaluator", FakeEvaluator)

    attack_db_path = tmp_path / "attack_baselines.sqlite"
    baseline_db_path = tmp_path / "baselines.sqlite"
    _write_fake_attack_cache(attack_db_path)
    _write_fake_baseline_cache(baseline_db_path)
    config = Exp3SequenceConfig(
        out_dir=tmp_path / "exp3_run",
        max_graphs=2,
        workers=1,
        trials_per_cell=2,
        baseline_cache_db_path=baseline_db_path,
        attack_cache_db_path=attack_db_path,
        attack_payoff_samples_per_route=2,
        online_turns=8,
        online_final_window=4,
        action_kinds=("memory_degradation",),
        security_monitor="chsh",
    )
    summary = run_pipeline(config, progress=False)

    out_dir = tmp_path / "exp3_run"
    assert summary["backend"] == "sequence_repeater_e91"
    assert summary["corpus"]["source"] == "sqlite"
    assert summary["corpus"]["metadata"]["max_route_hops"] == 7
    assert (out_dir / "config.json").exists()
    assert (out_dir / "corpus.json").exists()
    assert (out_dir / "baseline_health.json").exists()
    assert (out_dir / "oracle_summary.json").exists()
    assert (out_dir / "exp3_summary.json").exists()
    assert (out_dir / "dt" / "dt_payload.json").exists()
    assert (out_dir / "dt" / "graph_value_dt.json").exists()
    assert (out_dir / "dt" / "action_strategy_dt.json").exists()
    assert (out_dir / "dt" / "route_strategy_dt.json").exists()
    assert (out_dir / "figures" / "oracle_vs_exp3_retention.png").exists()
    assert (out_dir / "figures" / "selected_graph_exp3_convergence.png").exists()
    assert (out_dir / "figures" / "attack_chsh_qber_by_hop.png").exists()

    oracle = json.loads((out_dir / "oracle_summary.json").read_text())
    assert "disjoint_parallel_2_v0" in oracle
    assert oracle["disjoint_parallel_2_v0"]["retention"] == pytest.approx(0.5, abs=1e-6)

    exp3_payload = json.loads((out_dir / "exp3_summary.json").read_text())
    first_curve = exp3_payload["disjoint_parallel_2_v0"]["exp3_vs_exp3"][
        "learning_curve"
    ][0]
    run_metadata = exp3_payload["disjoint_parallel_2_v0"]["exp3_vs_exp3"][
        "metadata"
    ]
    assert first_curve["oracle_retention"] == pytest.approx(0.5, abs=1e-6)
    assert "retention_so_far" in first_curve
    assert "matrix_retention" in first_curve
    assert first_curve["gamma_A_t"] == pytest.approx(0.07)
    assert first_curve["eta_A_t"] == pytest.approx(0.07 / run_metadata["K_A"])
    assert "alice_policy_entropy" in first_curve
    assert "eve_policy_entropy" in first_curve
    assert first_curve["empirical_nash_gap"] == pytest.approx(first_curve["exploitability"])
    assert run_metadata["schedule_mode"] == "constant"
    assert run_metadata["alice_acceptance_rule"] == "chsh_only"
    assert run_metadata["reward_convention"].startswith("reward maximization")

    payoff_payload = json.loads(
        (out_dir / "payoff_matrices" / "disjoint_parallel_2_v0.json").read_text()
    )
    assert payoff_payload["backend"] == "sequence_repeater_e91"
    assert payoff_payload["security_monitor"] == "chsh"
    assert payoff_payload["trials_per_cell"] == 2
    first_cell = payoff_payload["cells"][0]
    assert first_cell["seed_start"] <= first_cell["seed_end"]
    assert first_cell["chsh_s_count"] == 2
    assert first_cell["qber_count"] == 2
    assert first_cell["mean_sifted_count"] == pytest.approx(350.0)
    assert first_cell["sequence_timing"]["alice_acceptance_rule"] == "chsh_only"
    assert "no_attack_false_signal_rate" in first_cell["sequence_timing"]
    assert first_cell["sequence_timing"]["start_time_ps"] == 1
    attack_cells = [
        cell for cell in payoff_payload["cells"]
        if cell["sequence_timing"].get("payoff_model") == "attack_route_profile_cache"
    ]
    assert attack_cells
    assert attack_cells[0]["trial_count"] == 2
    assert attack_cells[0]["simulated_action_id"] == "attack_cache:memory_degradation"
    assert attack_cells[0]["sequence_timing"]["attack_cache_target_counts"]

    health_payload = json.loads((out_dir / "baseline_health.json").read_text())
    assert health_payload["status"] == "healthy"
    assert health_payload["failed_graph_count"] == 0
    assert health_payload["accepted_count"] > 0
    assert health_payload["qualified_count"] > 0

    action_dt = json.loads((out_dir / "dt" / "action_strategy_dt.json").read_text())
    assert "action_expected_denial_under_oracle_alice" in action_dt["trees"]
    assert "action_expected_denial_under_oracle_alice" in action_dt["rows"][0]
    assert "active_hit_mean_chsh_s" in action_dt["rows"][0]
    assert "cautious_control_eve_empirical_prob" in action_dt["rows"][0]

    dt_payload = json.loads((out_dir / "dt" / "dt_payload.json").read_text())
    assert "cautious_greedy_final_retention" in dt_payload["graph_rows"][0]
    assert "cautious_greedy_alice_empirical_prob" in dt_payload["route_rows"][0]

    corpus_payload = json.loads((out_dir / "corpus.json").read_text())
    assert corpus_payload["source"]["kind"] == "sqlite"
    assert corpus_payload["metadata"]["max_route_hops"] == 7
    assert corpus_payload["summary"]["max_route_hops"] <= 7


class CountingFakeEvaluator(FakeEvaluator):
    instances: list["CountingFakeEvaluator"] = []

    def __init__(self, ir_dict, routes, config):
        super().__init__(ir_dict, routes, config)
        self.calls = 0
        CountingFakeEvaluator.instances.append(self)

    def evaluate(self, route_index: int, action: ActionSpec, *, seed: int, trial_id: str):
        self.calls += 1
        return super().evaluate(route_index, action, seed=seed, trial_id=trial_id)


def test_payoff_uses_route_baselines_and_attack_cache_hit_cells(
        monkeypatch, tmp_path):
    import sequence_game.experiments.exp3_sequence.payoff as payoff_mod
    from sequence_game.experiments.exp3_sequence.backend import _action_hits_route
    from sequence_game.experiments.exp3_sequence.payoff import (
        estimate_payoff,
        plan_cell_tasks,
    )

    monkeypatch.setattr(payoff_mod, "SequenceRouteEvaluator", CountingFakeEvaluator)
    CountingFakeEvaluator.instances.clear()

    attack_db_path = tmp_path / "attack_baselines.sqlite"
    _write_fake_attack_cache(attack_db_path)
    config = Exp3SequenceConfig(
        workers=1,
        trials_per_cell=3,
        baseline_cache_db_path=tmp_path / "missing_baselines.sqlite",
        attack_cache_db_path=attack_db_path,
        attack_payoff_samples_per_route=2,
    )
    case = load_graph_cases_from_sqlite(CORPUS_SQLITE_PATH, limit=1)[0]
    actions = build_actions(case.routes, config.action_kinds)
    tasks, cell_to_task = plan_cell_tasks(
        case.routes, actions,
        trials_per_cell=config.trials_per_cell, seed_base=1000)

    payoff = estimate_payoff(case, actions, config, seed_base=1000)

    total_cells = len(case.routes) * len(actions)
    assert len(tasks) < total_cells
    total_calls = sum(ev.calls for ev in CountingFakeEvaluator.instances)
    assert total_calls == len(tasks) * config.trials_per_cell

    cells = {(cell.route_id, cell.action_id): cell for cell in payoff.cells}
    no_attack_index = list(payoff.action_ids).index("no_attack")
    for route_index, route in enumerate(case.routes):
        no_attack_cell = cells[(str(route["route_id"]), "no_attack")]
        assert no_attack_cell.simulated_action_id == "no_attack"
        for action_index, action in enumerate(actions):
            cell = cells[(str(route["route_id"]), action.action_id)]
            if _action_hits_route(action, route):
                assert cell.simulated_action_id == f"attack_cache:{action.attack_type}"
                assert cell.accepted_rate == 0.0
                assert cell.trial_count == config.attack_payoff_samples_per_route
                assert cell.sequence_timing["payoff_model"] == "attack_route_profile_cache"
                assert cell.sequence_timing["attack_cache_target_counts"]
                if action.attack_type == "edge_intercept_resend":
                    assert cell.chsh_abort_count == config.attack_payoff_samples_per_route
                    assert cell.qber_abort_count == 0
                    assert cell.mean_chsh_s == pytest.approx(-0.5)
                    assert cell.mean_qber == pytest.approx(0.55)
                if action.attack_type == "memory_degradation":
                    assert cell.qber_abort_count == config.attack_payoff_samples_per_route
                    assert cell.chsh_abort_count == 0
                    assert cell.mean_chsh_s == pytest.approx(1.2)
                    assert cell.mean_qber == pytest.approx(0.30)
            else:
                # Route-missing actions reuse the clean route baseline.
                assert cell.simulated_action_id == "no_attack"
                assert cell.seed_start == no_attack_cell.seed_start
                assert cell.accepted_rate == no_attack_cell.accepted_rate
                assert payoff.payoff[route_index, action_index] == (
                    payoff.payoff[route_index, no_attack_index]
                )


def test_plan_cell_tasks_caches_identical_route_fiber_lengths():
    from sequence_game.experiments.exp3_sequence.payoff import plan_cell_tasks

    routes = [
        {
            "route_id": "r0",
            "path": ["a", "x", "b"],
            "internal_nodes": ["x"],
            "edge_lengths_m": [400.0, 425.0],
            "total_length_m": 825.0,
            "hop_count": 2,
        },
        {
            "route_id": "r1",
            "path": ["a", "y", "b"],
            "internal_nodes": ["y"],
            "edge_lengths_m": [400.0, 425.0],
            "total_length_m": 825.0,
            "hop_count": 2,
        },
    ]
    actions = [
        ActionSpec("no_attack", "none", "", "no_attack"),
        ActionSpec("memory_degradation:x", "node", "x", "memory_degradation"),
        ActionSpec("memory_degradation:y", "node", "y", "memory_degradation"),
    ]

    tasks, cell_to_task = plan_cell_tasks(
        routes, actions, trials_per_cell=1, seed_base=42)

    assert len(tasks) == 1
    assert {
        cell_to_task[(route_index, action_index)]
        for route_index in range(len(routes))
        for action_index in range(len(actions))
    } == {0}


def test_pipeline_parallel_workers(monkeypatch, tmp_path):
    # Fork-context workers inherit the monkeypatched evaluator, so this
    # exercises the pooled payoff-cell and online-condition phases end to end.
    import sequence_game.experiments.exp3_sequence.health as health_mod
    import sequence_game.experiments.exp3_sequence.payoff as payoff_mod

    monkeypatch.setattr(health_mod, "SequenceRouteEvaluator", FakeEvaluator)
    monkeypatch.setattr(payoff_mod, "SequenceRouteEvaluator", FakeEvaluator)

    attack_db_path = tmp_path / "attack_baselines.sqlite"
    baseline_db_path = tmp_path / "baselines.sqlite"
    _write_fake_attack_cache(attack_db_path)
    _write_fake_baseline_cache(baseline_db_path)
    config = Exp3SequenceConfig(
        out_dir=tmp_path / "exp3_parallel",
        max_graphs=2,
        workers=2,
        trials_per_cell=1,
        baseline_cache_db_path=baseline_db_path,
        attack_cache_db_path=attack_db_path,
        attack_payoff_samples_per_route=1,
        online_turns=4,
        online_final_window=2,
        action_kinds=("memory_degradation",),
        security_monitor="chsh",
    )
    summary = run_pipeline(config, progress=False)

    assert summary["graph_count"] == 2
    exp3_summary = json.loads(
        (tmp_path / "exp3_parallel" / "exp3_summary.json").read_text()
    )
    for graph_id, by_condition in exp3_summary.items():
        assert set(by_condition) == {
            "exp3_vs_exp3",
            "oracle_eve_vs_exp3_alice",
            "exp3_eve_vs_oracle_alice",
            "exp3_eve_vs_cautious_greedy_alice",
            "oracle_vs_oracle",
        }
        for run in by_condition.values():
            assert run["turns"] == 4


def test_pipeline_stops_when_no_attack_health_fails(monkeypatch, tmp_path):
    import sequence_game.experiments.exp3_sequence.health as health_mod

    monkeypatch.setattr(health_mod, "SequenceRouteEvaluator", FailingHealthEvaluator)

    config = Exp3SequenceConfig(
        out_dir=tmp_path / "exp3_unhealthy",
        max_graphs=1,
        workers=1,
        trials_per_cell=1,
        baseline_cache_db_path=tmp_path / "missing_baselines.sqlite",
        online_turns=1,
        baseline_health_routes_per_graph=1,
        baseline_health_trials_per_route=1,
        action_kinds=("memory_degradation",),
        security_monitor="chsh",
    )
    with pytest.raises(RuntimeError, match="baseline no-attack health failed"):
        run_pipeline(config, progress=False)

    health_payload = json.loads(
        (tmp_path / "exp3_unhealthy" / "baseline_health.json").read_text()
    )
    assert health_payload["status"] == "failed"
    assert health_payload["failed_graph_count"] == 1
    assert health_payload["graphs"]["disjoint_parallel_2_v0"]["warnings"] == [
        "no_no_attack_acceptance",
        "no_qualified_no_attack_acceptance",
        "low_no_attack_acceptance_rate",
        "all_chsh_abort",
    ]
