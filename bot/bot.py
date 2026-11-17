"""Bot RLBot v5 (S1-19) : fait jouer notre acteur dans le VRAI jeu via le wrapper rlgym-rlbot.

Réutilise EXACTEMENT les mêmes AttentionObs et action parser qu'à l'entraînement (ARCHITECTURE §12).
Le modèle est lu dans bot/model.pt, produit par :  python scripts/export_bot.py --run runs/<run>
ou dans le fichier passé en argument (relatif à bot/), pour faire jouer une autre version : voir agent_v*.toml.

⚠️ Hors ligne uniquement (RLBot). Jamais en ranked.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from rlgym_rlbot import RLGymBot, RLGymBotConfig  # noqa: E402

from rlagent.env.actions import make_action_parser  # noqa: E402
from rlagent.env.obs import BALL_PRED_DIM, RACE_DIM, SELF_DIM, AttentionObs  # noqa: E402
from rlagent.env.shadow_sim import ShadowSim, init_rocketsim, patch_compat_cars  # noqa: E402
from rlagent.train import build_actor_from_model_cfg  # noqa: E402


class Agent:
    def __init__(self, path: Path, deterministic: bool = False):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.actor = build_actor_from_model_cfg(data["model_cfg"])
        self.actor.load_state_dict(data["actor"])
        self.actor.eval()
        self.deterministic = deterministic
        self.action_repeat = int(data.get("action_repeat", 8))
        # observations identiques à l'entraînement : avec prédiction de balle / course à la balle si le réseau a été
        # entraîné avec (les options s'ajoutent à la fin du bloc « soi », cf. models/attention.py:SELF_DIMS)
        sd = int(data["model_cfg"].get("self_dim", SELF_DIM))
        self.ball_prediction = sd > SELF_DIM
        self.race_features = sd > SELF_DIM + BALL_PRED_DIM
        self.boost_features = sd > SELF_DIM + BALL_PRED_DIM + RACE_DIM
        torch.set_num_threads(1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.actor.act(x, deterministic=self.deterministic)
        return np.array([int(action.item())])


CAR_FLAGS = ("on_ground", "is_jumping", "has_jumped", "has_flipped", "is_flipping", "has_double_jumped", "can_flip",
             "has_flip", "air_time_since_jump", "jump_time", "handbrake", "is_holding_jump", "is_boosting", "flip_time")


def _v(v) -> list:
    return [v.x, v.y, v.z]


class Tracer:
    """Journal tick par tick (2e argument de bot.py) : état du paquet RLBot et indicateurs que le bot en déduit
    (rlgym_compat), pour les comparer à RocketSim (scripts/real_flags_check.py)."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")
        self.n = 0

    def record(self, packet, state) -> None:
        players = []
        for p in packet.players:
            ph, li = p.physics, p.last_input
            players.append({"team": p.team, "pos": _v(ph.location), "rot": [ph.rotation.pitch, ph.rotation.yaw,
                            ph.rotation.roll], "vel": _v(ph.velocity), "ang": _v(ph.angular_velocity),
                            "boost": p.boost, "air": int(p.air_state), "hj": p.has_jumped, "hdj": p.has_double_jumped,
                            "hd": p.has_dodged, "dto": p.dodge_timeout, "de": p.dodge_elapsed,
                            "demo": p.demolished_timeout,
                            "in": [li.throttle, li.steer, li.pitch, li.yaw, li.roll, li.jump, li.boost, li.handbrake]})
        cars = {str(c.team_num): [float(getattr(c, k)) for k in CAR_FLAGS] for c in state.cars.values()}
        self.f.write(json.dumps({"f": packet.match_info.frame_num, "phase": int(packet.match_info.match_phase),
                                 "p": players, "c": cars}) + "\n")
        self.n += 1
        if self.n % 240 == 0:
            self.f.flush()


class RLAgentBot(RLGymBot):
    def __init__(self):
        # argmax : mêmes poids, sans le bruit de l'échantillonnage (s10, 04/11 : 24-8-8 contre la version échantillonnée)
        self.agent = Agent(HERE / (sys.argv[1] if len(sys.argv) > 1 else "model.pt"), deterministic=True)
        super().__init__(
            obs_builder=AttentionObs(ball_prediction=self.agent.ball_prediction,
                                     race_features=self.agent.race_features,
                                     boost_features=self.agent.boost_features),
            action_parser=make_action_parser(self.agent.action_repeat),
            default_agent_id="ensisa/rlagent/v0.1",
            # Valeurs par défaut du wrapper = RocketSimEngine(rlbot_delay=True), comme à l'entraînement.
            config=RLGymBotConfig(),
        )

    def initialize(self):
        super().initialize()
        # Copie RocketSim recalée sur le jeu (env/shadow_sim.py) : jump_time, air_time_since_jump et is_boosting
        # tels qu'à l'entraînement (rlgym_compat les reconstruit mal). RLAGENT_NO_SHADOW=1 pour s'en passer.
        shadow = None
        if os.environ.get("RLAGENT_NO_SHADOW") != "1":
            init_rocketsim(HERE.parent / "collision_meshes")
            shadow = ShadowSim()
        tracer = Tracer(HERE / sys.argv[2]) if len(sys.argv) > 2 else None
        if shadow is None and tracer is None:
            return
        # GameState a des __slots__ : on enveloppe la méthode de la classe (ce processus seulement)
        cls = type(self.latest_game_state)
        update = cls.update

        def wrapped(gs, packet, extra_info=None):
            update(gs, packet, extra_info)
            if gs is not self.latest_game_state:
                return
            if shadow is not None:
                patch_compat_cars(gs, packet, shadow.update(packet))
            if tracer is not None:
                tracer.record(packet, gs)

        cls.update = wrapped

    def get_action(self, obs, game_state, packet):
        return self.agent.act(obs)


if __name__ == "__main__":
    RLAgentBot().run()
