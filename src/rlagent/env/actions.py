"""Espace d'actions : table discrète de 90 actions (standard RLGym) + mémorisation de l'action précédente.

L'action précédente est rangée dans shared_info["prev_actions"] pour que l'ObsBuilder la voie
(RLGym appelle parse_actions avant build_obs à chaque step, et reset de l'action parser avant build_obs au reset).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import GameState


class TrackedLookupTableAction(LookupTableAction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        super().reset(agents, initial_state, shared_info)
        shared_info["prev_actions"] = {a: np.zeros(8, dtype=np.float32) for a in agents}

    def parse_actions(self, actions: Dict[str, np.ndarray], state: GameState,
                      shared_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        parsed = super().parse_actions(actions, state, shared_info)
        prev = shared_info.setdefault("prev_actions", {})
        for agent, controls in parsed.items():
            prev[agent] = np.asarray(controls, dtype=np.float32).reshape(-1, 8)[-1].copy()
        return parsed


def make_action_parser(repeats: int) -> RepeatAction:
    return RepeatAction(TrackedLookupTableAction(), repeats=repeats)


N_ACTIONS = len(LookupTableAction.make_lookup_table())
