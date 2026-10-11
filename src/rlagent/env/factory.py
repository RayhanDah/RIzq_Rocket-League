"""Construction de l'environnement RLGym v2 + RocketSim à partir de la config."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.state_mutators import MutatorSequence

from .actions import make_action_parser
from .mutators import RandomFormatMutator, StartStateMutator
from .obs import AttentionObs
from .rewards import make_reward

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SOCCAR_MESHES = 16  # nombre habituel de fichiers soccar produits par le dumper


class MeshesMissingError(RuntimeError):
    pass


def resolve_meshes_dir(cfg: Dict[str, Any]) -> Path:
    raw = Path(os.environ.get("RS_COLLISION_MESHES") or cfg["paths"]["collision_meshes"])
    if not raw.is_absolute():
        cwd_candidate = (Path.cwd() / raw)
        raw = cwd_candidate if cwd_candidate.exists() else REPO_ROOT / raw
    return raw.resolve()


def check_meshes(meshes_dir: Path) -> int:
    soccar = meshes_dir / "soccar"
    files = sorted(soccar.glob("*.cmf")) if soccar.is_dir() else []
    if not files:
        raise MeshesMissingError(
            f"Aucun fichier de collision trouvé dans {soccar}.\n"
            "RocketSim ne peut PAS fonctionner sans eux (il fermerait le processus sans message clair).\n"
            "→ Lancer Rocket League en free play (stade standard), puis RLArenaCollisionDumper.exe,\n"
            "  et copier le dossier produit `collision_meshes/` (contenant `soccar/mesh_*.cmf`) à la racine du projet.\n"
            "  Voir INSTALL.md §4."
        )
    return len(files)


_RS_INITED = False


def init_rocketsim(cfg: Dict[str, Any]) -> Path:
    """Initialise RocketSim AVANT toute création d'arène, après avoir vérifié les meshes."""
    global _RS_INITED
    meshes_dir = resolve_meshes_dir(cfg)
    check_meshes(meshes_dir)
    if not _RS_INITED:
        import RocketSim as rsim  # import tardif : les tests purs n'en ont pas besoin
        try:
            rsim.init(str(meshes_dir))
        except RuntimeError as err:  # "Already inited" : déjà fait dans ce processus
            if "inited" not in str(err).lower():
                raise
        _RS_INITED = True
    return meshes_dir


def build_env(cfg: Dict[str, Any], seed: Optional[int] = None) -> RLGym:
    init_rocketsim(cfg)
    from rlgym.rocket_league.sim import RocketSimEngine

    e = cfg["env"]
    return RLGym(
        state_mutator=MutatorSequence(
            RandomFormatMutator(e["format_weights"], seed=seed),
            StartStateMutator(e["start_weights"], seed=None if seed is None else seed + 1),
        ),
        obs_builder=AttentionObs(),
        action_parser=make_action_parser(e["action_repeat"]),
        reward_fn=make_reward(cfg),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=e["no_touch_timeout_s"]),
            TimeoutCondition(timeout_seconds=e["episode_timeout_s"]),
        ),
        transition_engine=RocketSimEngine(rlbot_delay=e.get("rlbot_delay", True)),
    )
