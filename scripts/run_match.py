"""Lance un match dans le vrai jeu via RLBot v5 et suit le score. Hors ligne uniquement, jamais en ranked.

    .venv-bot\\Scripts\\python scripts\\run_match.py                        # notre agent contre Psyonix All-Star
    .venv-bot\\Scripts\\python scripts\\run_match.py --config bot/autre.toml

RLBotServer doit être installé dans %LOCALAPPDATA%\\RLBot5\\bin (INSTALL.md §8). À la fin du match, le jeu reste ouvert sur
l'écran des scores.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlbot import flat
from rlbot.config import load_match_config
from rlbot.managers import MatchManager

ROOT = Path(__file__).resolve().parents[1]


def score(packet: flat.GamePacket) -> tuple[int, int]:
    goals = {t.team_index: t.score for t in packet.teams}
    return goals.get(0, 0), goals.get(1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "bot" / "match_allstar.toml"))
    args = ap.parse_args()

    match = MatchManager()
    match.start_match(load_match_config(args.config))
    print("Match lancé.", flush=True)

    last = None
    while match.rlbot_interface.is_connected:
        packet = match.packet
        if packet is not None:
            now = score(packet)
            if now != last:
                left = int(packet.match_info.game_time_remaining)
                print(f"[{left // 60}:{left % 60:02d}] bleu {now[0]} - {now[1]} orange", flush=True)
                last = now
            if packet.match_info.match_phase == flat.MatchPhase.Ended:
                print(f"Score final : bleu {now[0]} - {now[1]} orange", flush=True)
                break
        time.sleep(0.5)
    match.disconnect()


if __name__ == "__main__":
    main()
