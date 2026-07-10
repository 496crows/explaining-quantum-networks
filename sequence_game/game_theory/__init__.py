"""Zero-sum game-theoretic solvers for the routing/interdiction game.

The Alice(routing)-vs-Eve(interdiction) interaction on a graph is a zero-sum
covering game: Alice packs key rate onto routes, Eve covers routes with node/edge
attacks. This package provides adversarial *co-learning* solvers (no baked-in
equilibrium) plus an exact minimax LP used only as a validation oracle.
"""

from .bandit import (
    ConvergenceTrajectory,
    Exp3,
    Exp3SchedulePoint,
    bandit_selfplay_trajectory,
    solve_zero_sum_bandit_selfplay,
)
from .zero_sum import (
    GameCharacterization,
    RegretMatchingPlus,
    characterize_equilibrium,
    exploitability,
    solve_zero_sum_lp,
    solve_zero_sum_selfplay,
)

__all__ = [
    "ConvergenceTrajectory",
    "Exp3",
    "Exp3SchedulePoint",
    "GameCharacterization",
    "RegretMatchingPlus",
    "bandit_selfplay_trajectory",
    "characterize_equilibrium",
    "exploitability",
    "solve_zero_sum_bandit_selfplay",
    "solve_zero_sum_lp",
    "solve_zero_sum_selfplay",
]
