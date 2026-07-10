"""Bandit no-regret co-learning for the asymmetric partial-information game.

Each round Alice picks a route and Eve picks an attack *simultaneously*, and each
player observes **only the realised outcome of their own move** -- never the
opponent's action:

* Alice sees her key-rate outcome (accepted / QBER-abort / CHSH-abort /
  delivery-failure), but not *where* on the route Eve struck.
* Eve sees only whether her move *hit a real signal* (the classical control
  record), not Alice's full route.

This is bandit feedback, so the honest co-learner is Exp3 (exponential weights
with importance-weighted estimates), not the full-information RM+ solver. Two
Exp3 learners in a zero-sum game are each no-regret, so their empirical play
converges to the minimax equilibrium -- discovered adversarially, never baked in.
The full-information LP/RM+ solver (:mod:`.zero_sum`) is used only as the oracle
to check what the bandit co-learners converge to.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .zero_sum import exploitability

EXP3_SCHEDULE_CONSTANT = "constant"
EXP3_SCHEDULE_ANYTIME = "anytime"
EXP3_SCHEDULE_MODES = frozenset({
    EXP3_SCHEDULE_CONSTANT,
    EXP3_SCHEDULE_ANYTIME,
})


@dataclass(frozen=True)
class Exp3SchedulePoint:
    turn: int
    eta: float
    gamma: float


class Exp3:
    """Exp3 adversarial-bandit learner maximizing rewards in [0, 1].

    Constant mode preserves the historical implementation exactly: action
    selection uses the fixed exploration mass ``gamma`` and the log-weight
    update uses ``eta = gamma / K``. Anytime mode keeps the same reward
    maximization convention but uses the supplied player-specific schedule.
    """

    def __init__(self, num_actions: int, *, gamma: float = 0.07,
                 rng: np.random.Generator | None = None,
                 schedule_mode: str = EXP3_SCHEDULE_CONSTANT,
                 eta_c: float = 1.0,
                 t0: float = 10000.0,
                 gamma_max: float = 0.20):
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if schedule_mode not in EXP3_SCHEDULE_MODES:
            raise ValueError(f"unsupported Exp3 schedule mode {schedule_mode!r}")
        if eta_c <= 0:
            raise ValueError("eta_c must be > 0")
        if t0 < 0:
            raise ValueError("t0 must be >= 0")
        if not 0 < gamma_max <= 1:
            raise ValueError("gamma_max must be in (0, 1]")
        self.k = num_actions
        self.gamma = gamma
        self.schedule_mode = schedule_mode
        self.eta_c = float(eta_c)
        self.t0 = float(t0)
        self.gamma_max = float(gamma_max)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._logw = np.zeros(num_actions)     # log-weights (overflow-safe)
        self.counts = np.zeros(num_actions)     # empirical action frequency
        self._last_p: np.ndarray | None = None
        self._last_a: int | None = None
        self._last_schedule: Exp3SchedulePoint | None = None

    def schedule_at_turn(self, turn: int) -> Exp3SchedulePoint:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        if self.schedule_mode == EXP3_SCHEDULE_CONSTANT:
            return Exp3SchedulePoint(
                turn=int(turn),
                eta=float(self.gamma / self.k),
                gamma=float(self.gamma),
            )
        eta = self.eta_c * math.sqrt(
            math.log(self.k) / (self.k * (float(turn) + self.t0))
        )
        gamma = min(self.gamma_max, self.k * eta)
        return Exp3SchedulePoint(turn=int(turn), eta=float(eta), gamma=float(gamma))

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        point = self.schedule_at_turn(
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        w = np.exp(self._logw - self._logw.max())
        w /= w.sum()
        p = (1.0 - point.gamma) * w + point.gamma / self.k
        total = float(p.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("invalid Exp3 policy normalization")
        p = p / total
        floor = point.gamma / self.k
        if np.any(p < floor - 1e-15):
            raise FloatingPointError("Exp3 policy fell below exploration floor")
        return p

    def sample(self) -> int:
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        p = self.policy(turn=turn)
        a = int(self.rng.choice(self.k, p=p))
        self._last_p, self._last_a = p, a
        self._last_schedule = point
        self.counts[a] += 1
        return a

    def update(self, reward: float) -> None:
        """Update from the reward in [0, 1] of the last sampled action only."""
        if self._last_a is None:
            raise RuntimeError("update() called before sample()")
        if self._last_schedule is None or self._last_p is None:
            raise RuntimeError("update() called before sample()")
        a = self._last_a
        x_hat = float(reward) / self._last_p[a]      # importance-weighted estimate
        self._logw[a] += self._last_schedule.eta * x_hat

    def empirical_strategy(self) -> np.ndarray:
        total = self.counts.sum()
        if total > 0:
            return self.counts / total
        return np.full(self.k, 1.0 / self.k)

    @property
    def last_schedule(self) -> Exp3SchedulePoint | None:
        return self._last_schedule


@dataclass(frozen=True)
class BanditSelfPlayResult:
    value: float
    alice_strategy: np.ndarray     # Alice's empirical route distribution
    eve_strategy: np.ndarray       # Eve's empirical attack distribution
    iterations: int


def solve_zero_sum_bandit_selfplay(
        payoff: np.ndarray, *, iterations: int = 40000, gamma: float = 0.07,
        seed: int = 0) -> BanditSelfPlayResult:
    """Co-learn the equilibrium under bandit feedback (Exp3 self-play).

    ``payoff[a, e]`` is Alice's realised key rate for (route a, attack e). Alice's
    bandit reward is that rate; Eve's is the denial ``1 - rate`` (what she infers
    from hitting a real signal). Neither observes the other's action.
    """

    P = np.asarray(payoff, dtype=float)
    if P.ndim != 2:
        raise ValueError("payoff must be a 2D matrix")
    na, ne = P.shape
    rng = np.random.default_rng(seed)
    alice = Exp3(na, gamma=gamma, rng=rng)
    eve = Exp3(ne, gamma=gamma, rng=rng)
    for _ in range(iterations):
        a = alice.sample()
        e = eve.sample()
        rate = P[a, e]
        alice.update(rate)              # Alice: only her key-rate outcome
        eve.update(1.0 - rate)          # Eve: only "hit a real signal" -> denial
    x = alice.empirical_strategy()
    y = eve.empirical_strategy()
    return BanditSelfPlayResult(
        value=float(x @ P @ y),
        alice_strategy=x,
        eve_strategy=y,
        iterations=iterations,
    )


@dataclass(frozen=True)
class ConvergenceTrajectory:
    """Exploitability (Nash gap) of the running average vs iteration.

    The exploitability decaying to 0 is the no-regret convergence proof: when the
    average profile's Nash gap is ``eps`` it is an ``eps``-equilibrium, so
    ``exploitability -> 0`` certifies convergence to the minimax/Nash value (which
    ``value`` approaches). Provided for the learning-curve plots.
    """

    iters: np.ndarray            # checkpoint iteration counts
    value: np.ndarray            # running-average game value at each checkpoint
    exploitability: np.ndarray   # Nash gap of the running average at each checkpoint
    alice_strategy: np.ndarray
    eve_strategy: np.ndarray


def bandit_selfplay_trajectory(
        payoff: np.ndarray, *, iterations: int = 40000, record_every: int = 500,
        gamma: float = 0.07, seed: int = 0) -> ConvergenceTrajectory:
    """Exp3 self-play recording the exploitability trajectory (convergence proof)."""

    P = np.asarray(payoff, dtype=float)
    na, ne = P.shape
    rng = np.random.default_rng(seed)
    alice = Exp3(na, gamma=gamma, rng=rng)
    eve = Exp3(ne, gamma=gamma, rng=rng)
    iters, values, gaps = [], [], []
    for t in range(iterations):
        a = alice.sample()
        e = eve.sample()
        rate = P[a, e]
        alice.update(rate)
        eve.update(1.0 - rate)
        if (t + 1) % record_every == 0:
            x = alice.empirical_strategy()
            y = eve.empirical_strategy()
            iters.append(t + 1)
            values.append(float(x @ P @ y))
            gaps.append(exploitability(P, x, y))
    return ConvergenceTrajectory(
        iters=np.array(iters), value=np.array(values),
        exploitability=np.array(gaps),
        alice_strategy=alice.empirical_strategy(),
        eve_strategy=eve.empirical_strategy())
