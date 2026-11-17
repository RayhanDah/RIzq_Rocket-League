"""Copie RocketSim recalée sur le vrai jeu : les compteurs internes de la voiture tels qu'à l'entraînement.

Dans le vrai jeu, rlgym_compat reconstruit l'état de la voiture à partir du paquet RLBot. Trois valeurs vues par
le réseau y sont fausses une bonne partie du temps (scripts/real_flags_check.py, match du 14/11/2026, 37 500 ticks
par voiture) :
  - jump_time n'est jamais remis à 0 au sol (44-49 % des ticks) ;
  - air_time_since_jump se fige après un flip, quand dodge_timeout vaut -1 (41-44 %) ;
  - is_boosting est vrai quand on appuie sur boost avec le réservoir vide (15-33 %).

Ici, chaque voiture a son double dans une arène RocketSim (sans balle ni contacts entre voitures). À chaque paquet :
on applique au double les entrées réellement jouées depuis le paquet précédent, on avance du nombre de ticks
écoulés, on lit ses compteurs, puis on recale sa physique (position, rotation, vitesses, boost) sur le paquet.
Les booléens que le jeu donne lui-même (has_jumped, has_dodged…) restent ceux du paquet.

Le paquet est lu par attributs (packet.players[i].physics.location.x…) : marche avec les flatbuffers RLBot comme
avec un objet qui en a la forme (tests).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

ACTIVE_PHASES = (2, 3)             # MatchPhase Kickoff, Active : la physique tourne et les entrées comptent
TELEPORT_UU = 300.0                # déplacement en un tick au-delà duquel la voiture a été replacée (remise en jeu)


def init_rocketsim(meshes_dir: Path) -> None:
    import RocketSim as rs

    try:
        rs.init(str(meshes_dir))
    except RuntimeError as err:    # « Already inited » : déjà fait dans ce processus
        if "inited" not in str(err).lower():
            raise


class ShadowSim:
    def __init__(self):
        import RocketSim as rs

        self.rs = rs
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.arena.set_car_ball_collision(False)
        self.arena.set_car_car_collision(False)
        self.cars: Dict[int, object] = {}            # player_id → voiture RocketSim
        self.prev_pos: Dict[int, Optional[list]] = {}
        self.frame: Optional[int] = None

    def _sync(self, st, p) -> None:
        rs, ph = self.rs, p.physics
        st.pos = rs.Vec(ph.location.x, ph.location.y, ph.location.z)
        st.vel = rs.Vec(ph.velocity.x, ph.velocity.y, ph.velocity.z)
        st.ang_vel = rs.Vec(ph.angular_velocity.x, ph.angular_velocity.y, ph.angular_velocity.z)
        st.rot_mat = rs.Angle(yaw=ph.rotation.yaw, pitch=ph.rotation.pitch, roll=ph.rotation.roll).as_rot_mat()
        st.boost = float(p.boost)

    def _controls(self, li):
        return self.rs.CarControls(throttle=li.throttle, steer=li.steer, pitch=li.pitch, yaw=li.yaw, roll=li.roll,
                                   jump=bool(li.jump), boost=bool(li.boost), handbrake=bool(li.handbrake))

    def update(self, packet) -> Dict[int, object]:
        """Avance les doubles jusqu'à ce paquet ; renvoie {player_id: CarState RocketSim} (voitures suivies)."""
        frame = int(packet.match_info.frame_num)
        active = int(packet.match_info.match_phase) in ACTIVE_PHASES
        gap = 0 if self.frame is None else max(0, frame - self.frame)
        self.frame = frame
        players = list(packet.players)
        ids = {p.player_id for p in players}
        for pid in [k for k in self.cars if k not in ids]:
            self.arena.remove_car(self.cars.pop(pid))
            self.prev_pos.pop(pid, None)
        # 1) entrées jouées entre le paquet précédent et celui-ci (celles que porte ce paquet, cf. real_flags_check)
        stepping = []
        for p in players:
            car = self.cars.get(p.player_id)
            if car is not None and active and gap > 0 and p.demolished_timeout == -1:
                car.set_controls(self._controls(p.last_input))
                stepping.append(p.player_id)
            elif car is not None:
                car.set_controls(self.rs.CarControls())
        if stepping and gap > 0:
            self.arena.step(gap)
        # 2) recalage de la physique sur le paquet, état interne repris de zéro si la voiture a été replacée
        out = {}
        for p in players:
            car = self.cars.get(p.player_id)
            if car is None:
                car = self.arena.add_car(int(p.team), self.rs.CarConfig.OCTANE)
                self.cars[p.player_id] = car
                self.prev_pos[p.player_id] = None
            loc = p.physics.location
            pos = [loc.x, loc.y, loc.z]
            prev = self.prev_pos.get(p.player_id)
            moved = prev is None or sum((a - b) ** 2 for a, b in zip(pos, prev)) > TELEPORT_UU ** 2
            reset = moved or not active or p.demolished_timeout != -1
            st = self.rs.CarState() if reset else car.get_state()
            self._sync(st, p)
            car.set_state(st)
            self.prev_pos[p.player_id] = pos
            if p.player_id in stepping and not moved:
                out[p.player_id] = st
        return out


def patch_compat_cars(game_state, packet, states: Dict[int, object]) -> None:
    """Corrige dans l'état rlgym_compat (clés = player_id) les trois compteurs faux, depuis les doubles RocketSim."""
    by_id = {p.player_id: p for p in packet.players}
    for pid, st in states.items():
        car = game_state.cars.get(pid)
        p = by_id.get(pid)
        if car is None or p is None:
            continue
        car.jump_time = st.jump_time
        if p.dodge_timeout == -1:            # sinon le paquet donne la valeur exacte (1,25 - dodge_timeout)
            car.air_time_since_jump = st.air_time_since_jump
        car.boost_active_time = st.time_spent_boosting
