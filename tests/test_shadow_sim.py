"""Copie RocketSim du bot (env/shadow_sim.py) : retrouve les compteurs d'une voiture de référence à partir des seuls
paquets (physique + entrées), comme dans le vrai jeu."""
from types import SimpleNamespace as NS

import pytest

rs = pytest.importorskip("RocketSim")

from rlagent.config import load_config  # noqa: E402
from rlagent.env.factory import MeshesMissingError, init_rocketsim  # noqa: E402
from rlagent.env.shadow_sim import ShadowSim, patch_compat_cars  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _meshes():
    try:
        init_rocketsim(load_config([]))
    except MeshesMissingError:
        pytest.skip("collision meshes absents")


def vec(v):
    return NS(x=v.x, y=v.y, z=v.z)


def packet(frame, car, controls, boost):
    st = car.get_state()
    a = st.rot_mat.as_angle()
    li = NS(throttle=controls.throttle, steer=controls.steer, pitch=controls.pitch, yaw=controls.yaw,
            roll=controls.roll, jump=controls.jump, boost=controls.boost, handbrake=controls.handbrake)
    p = NS(player_id=7, team=0, boost=boost, demolished_timeout=-1, dodge_timeout=-1, last_input=li,
           physics=NS(location=vec(st.pos), velocity=vec(st.vel), angular_velocity=vec(st.ang_vel),
                      rotation=NS(pitch=a.pitch, yaw=a.yaw, roll=a.roll)))
    return NS(match_info=NS(frame_num=frame, match_phase=3), players=[p])


def script():
    """(ticks, contrôles) : rouler, sauter, retomber, rouler, sauter + flip, boost réservoir vide au sol."""
    C = rs.CarControls
    return [(30, C(throttle=1.0)), (12, C(throttle=1.0, jump=True)), (100, C(throttle=1.0)),
            (40, C(throttle=1.0)), (10, C(jump=True)), (6, C()), (2, C(jump=True, pitch=-1.0)),
            (150, C(throttle=1.0)), (60, C(throttle=1.0, boost=True))]


def test_shadow_recovers_reference_counters():
    arena = rs.Arena(rs.GameMode.SOCCAR)
    arena.set_car_ball_collision(False)
    ref = arena.add_car(0, rs.CarConfig.OCTANE)
    st = ref.get_state()
    st.pos, st.boost = rs.Vec(0.0, -2000.0, 17.0), 6.0       # réservoir presque vide : vide pendant le boost final
    ref.set_state(st)
    arena.step(30)                                            # posé au sol
    shadow, frame = ShadowSim(), 100
    ctrl = rs.CarControls()
    shadow.update(packet(frame, ref, ctrl, ref.get_state().boost))
    checked = {"jump_time": 0, "air_time_since_jump": 0, "is_boosting": 0}
    for ticks, ctrl in script():
        if ctrl.boost:                                        # réservoir vidé (sinon les pastilles le remplissent)
            st = ref.get_state()
            st.boost = 0.0
            ref.set_state(st)
            shadow.update(packet(frame, ref, ctrl, 0.0))      # même frame : recalage sans avancer
        ref.set_controls(ctrl)
        for _ in range(ticks):
            arena.step(1)
            frame += 1
            want = ref.get_state()
            got = shadow.update(packet(frame, ref, ctrl, want.boost)).get(7)
            assert got is not None
            assert got.jump_time == pytest.approx(want.jump_time, abs=1e-5)
            assert got.air_time_since_jump == pytest.approx(want.air_time_since_jump, abs=1e-5)
            assert (got.time_spent_boosting > 0) == (want.time_spent_boosting > 0)
            checked["jump_time"] += want.jump_time > 0
            checked["air_time_since_jump"] += want.air_time_since_jump > 0
            checked["is_boosting"] += ctrl.boost and want.boost < 1.0
    # le script passe bien par les cas fautifs de rlgym_compat (saut puis sol, vol après flip, boost à vide)
    assert all(v > 5 for v in checked.values()), checked


def test_patch_overrides_only_the_three_counters():
    car = NS(jump_time=0.9, air_time_since_jump=0.05, boost_active_time=0.3, has_jumped=True)
    st = NS(jump_time=0.0, air_time_since_jump=0.7, time_spent_boosting=0.0)
    p = NS(player_id=3, dodge_timeout=-1)
    patch_compat_cars(NS(cars={3: car}), NS(players=[p]), {3: st})
    assert (car.jump_time, car.air_time_since_jump, car.boost_active_time, car.has_jumped) == (0.0, 0.7, 0.0, True)
    car.air_time_since_jump, p.dodge_timeout = 0.2, 1.05        # le paquet donne la valeur exacte : on la garde
    patch_compat_cars(NS(cars={3: car}), NS(players=[p]), {3: st})
    assert car.air_time_since_jump == 0.2


def test_countdown_resets_counters_like_a_training_reset():
    """1re décision du kickoff, prise pendant le compte à rebours : compteurs à zéro (diagnostic du 25/11)."""
    arena = rs.Arena(rs.GameMode.SOCCAR)
    arena.set_car_ball_collision(False)
    ref = arena.add_car(0, rs.CarConfig.OCTANE)
    shadow, frame = ShadowSim(), 100
    jump = rs.CarControls(throttle=1.0, jump=True, boost=True)
    ref.set_controls(jump)
    for _ in range(20):                                       # en l'air, boost : compteurs non nuls
        arena.step(1)
        frame += 1
        shadow.update(packet(frame, ref, jump, 50.0))
    st = ref.get_state()                                      # remise en place pour le kickoff, puis compte à rebours
    st.pos, st.vel, st.ang_vel = rs.Vec(0.0, -4608.0, 17.0), rs.Vec(0, 0, 0), rs.Vec(0, 0, 0)
    ref.set_state(st)
    countdown = packet(frame + 400, ref, rs.CarControls(), 33.3)
    countdown.match_info.match_phase = 1
    got = shadow.update(countdown).get(7)
    assert got is not None and got.jump_time == 0.0 and got.time_spent_boosting == 0.0
    car = NS(jump_time=1.0, air_time_since_jump=0.4, boost_active_time=0.5)   # valeurs périmées de rlgym_compat
    patch_compat_cars(NS(cars={7: car}), countdown, {7: got})
    assert (car.jump_time, car.air_time_since_jump, car.boost_active_time) == (0.0, 0.0, 0.0)
