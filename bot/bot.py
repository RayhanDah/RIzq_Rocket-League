"""Bot RLBot v5 (S1-19) : fait jouer notre acteur dans le VRAI jeu via le wrapper rlgym-rlbot.

Réutilise EXACTEMENT les mêmes AttentionObs et action parser qu'à l'entraînement (ARCHITECTURE §12).
Le modèle est lu dans bot/model.pt, produit par :  python scripts/export_bot.py --run runs/<run>

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
from rlagent.env.obs import AttentionObs  # noqa: E402
from rlagent.train import build_actor_from_model_cfg  # noqa: E402


class Agent:
    def __init__(self, path: Path, deterministic: bool = False):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.actor = build_actor_from_model_cfg(data["model_cfg"])
        self.actor.load_state_dict(data["actor"])
        self.actor.eval()
        self.deterministic = deterministic
        self.action_repeat = int(data.get("action_repeat", 8))
        torch.set_num_threads(1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.actor.act(x, deterministic=self.deterministic)
        return np.array([int(action.item())])


class RLAgentBot(RLGymBot):
    def __init__(self):
        self.agent = Agent(HERE / "model.pt")
        super().__init__(
            obs_builder=AttentionObs(),
            action_parser=make_action_parser(self.agent.action_repeat),
            default_agent_id="ensisa/rlagent/v0.1",
            # Valeurs par défaut du wrapper = RocketSimEngine(rlbot_delay=True), comme à l'entraînement.
            config=RLGymBotConfig(),
        )

    def get_action(self, obs, game_state, packet):
        return self.agent.act(obs)


if __name__ == "__main__":
    RLAgentBot().run()
