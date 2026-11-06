"""Bot RLBot v5 (S1-19) : fait jouer notre acteur dans le VRAI jeu via le wrapper rlgym-rlbot.

Réutilise EXACTEMENT les mêmes AttentionObs et action parser qu'à l'entraînement (ARCHITECTURE §12).
Le modèle est lu dans bot/model.pt, produit par :  python scripts/export_bot.py --run runs/<run>
ou dans le fichier passé en argument (relatif à bot/), pour faire jouer une autre version : voir agent_v*.toml.

⚠️ Hors ligne uniquement (RLBot). Jamais en ranked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from rlgym_rlbot import RLGymBot, RLGymBotConfig  # noqa: E402

from rlagent.env.actions import make_action_parser  # noqa: E402
from rlagent.env.obs import BALL_PRED_DIM, RACE_DIM, SELF_DIM, AttentionObs  # noqa: E402
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

    def get_action(self, obs, game_state, packet):
        return self.agent.act(obs)


if __name__ == "__main__":
    RLAgentBot().run()
