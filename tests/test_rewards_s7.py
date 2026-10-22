"""s7 : prédictions (tirs cadrés, ETA), rewards potentiels et de compétence, features « course à la balle »,
exercices à niveaux + curriculum, lot progressif."""
import numpy as np
import pytest
from rlgym.rocket_league import common_values as cv

from helpers import make_state, mirror_car_into
from rlagent.config import load_config
from rlagent.env.predict import ball_z_at, car_eta, compute_race, goal_entry_time, goal_threat, race_info
from rlagent.env.rewards import (AerialTouchTerm, BallGoalPotential, BoostPotential, ConcedeTerm, KickoffRushTerm,
                                 KickoffWinTerm, PositioningPotential, ThreatPotential, make_reward)

R = cv.BALL_RADIUS


def _set_ball(s, pos, vel=(0, 0, 0)):
    s.ball.position = np.array(pos, dtype=np.float32)
    s.ball.linear_velocity = np.array(vel, dtype=np.float32)
    s.ball.angular_velocity = np.zeros(3, dtype=np.float32)


def _place(s, agent, pos, yaw=0.0, vel=(0, 0, 0), boost=33.0):
    c = s.cars[agent]
    c.physics.position = np.array(pos, dtype=np.float32)
    c.physics.linear_velocity = np.array(vel, dtype=np.float32)
    c.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    c.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
    c.boost_amount = boost


def _mirror_state(s):
    """Image de l'état par rotation de 180° autour de Z, bleu ↔ orange."""
    m = make_state(len(s.cars) // 2, kickoff=True)
    m.tick_count = s.tick_count
    m.ball.position = s.ball.position * np.array([-1, -1, 1], dtype=np.float32)
    m.ball.linear_velocity = s.ball.linear_velocity * np.array([-1, -1, 1], dtype=np.float32)
    m.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    for a in s.cars:
        b = a.replace("blue", "tmp").replace("orange", "blue").replace("tmp", "orange")
        mirror_car_into(s, a, m, b)
    return m


def _swap(a):
    return a.replace("blue", "tmp").replace("orange", "blue").replace("tmp", "orange")


# --------------------------------------------------------------------------- prédictions
def test_goal_entry_detects_on_target_shots_only():
    t = goal_entry_time([0, 3000, R], [0, 2000, 0], 1.0)
    assert t is not None and 1.0 < t < 1.2                      # ~2120 uu à 2000 uu/s (frottement compris)
    assert goal_entry_time([0, 3000, R], [1500, 2000, 0], 1.0) is None     # à côté du poteau
    assert goal_entry_time([0, 3000, R], [0, 2000, 1300], 1.0) is None     # au-dessus de la barre
    assert goal_entry_time([0, 3000, R], [0, -2000, 0], 1.0) is None       # part dans l'autre sens
    assert goal_entry_time([0, -1000, R], [0, 300, 0], 1.0) is None        # trop lent : arrive après 3 s
    assert goal_entry_time([0, cv.BACK_WALL_Y, 300], [0, 0, 0], 1.0) == 0.0   # déjà sur la ligne
    assert goal_threat([0, 3000, R], [0, 2000, 0], 1.0) > goal_threat([0, 1000, R], [0, 2000, 0], 1.0) > 0
    # rebond sur le mur latéral qui revient dans le cadre
    assert goal_entry_time([3000, 2000, R], [2100, 1500, 0], 1.0) is not None


def test_ball_height_with_bounces():
    assert np.isclose(ball_z_at(1000, 0, 0.5), 1000 - 0.5 * 650 * 0.25)
    t_land = np.sqrt(2 * (1000 - R) / 650)
    assert np.isclose(ball_z_at(1000, 0, t_land), R, atol=1.0)
    assert ball_z_at(1000, 0, t_land + 0.3) > R + 100               # rebondit
    assert ball_z_at(R, 0, 2.0) == R                                # roule


def test_eta_orders_cars_sensibly():
    s = make_state(1, kickoff=False)
    _set_ball(s, [0, 0, R])
    pos, vel = s.ball.position.tolist(), s.ball.linear_velocity.tolist()
    _place(s, "blue-0", [0, -1000, 17], yaw=np.pi / 2)              # face à la balle
    near = car_eta(s.cars["blue-0"], pos, vel)[0]
    _place(s, "blue-0", [0, -3000, 17], yaw=np.pi / 2)
    far = car_eta(s.cars["blue-0"], pos, vel)[0]
    _place(s, "blue-0", [0, -3000, 17], yaw=-np.pi / 2)             # dos à la balle
    back = car_eta(s.cars["blue-0"], pos, vel)[0]
    _place(s, "blue-0", [0, -3000, 17], yaw=np.pi / 2, boost=0.0)
    no_boost = car_eta(s.cars["blue-0"], pos, vel)[0]
    _place(s, "blue-0", [0, -3000, 17], yaw=np.pi / 2, vel=(0, 2000, 0))
    fast = car_eta(s.cars["blue-0"], pos, vel)[0]
    assert near < far < back and far < no_boost and fast < far
    _set_ball(s, [0, 0, 1200])                                      # balle haute : plus long à atteindre
    _place(s, "blue-0", [0, -1000, 17], yaw=np.pi / 2)
    assert car_eta(s.cars["blue-0"], s.ball.position.tolist(), [0, 0, 0])[0] > near


def test_eta_continuous_at_takeoff():
    """Sauter ne fait pas bondir l'ETA (sinon `positioning` punit chaque saut et devient bruité) : en l'air, la voiture
    garde sa vitesse puis réaccélère au sol."""
    s = make_state(1, kickoff=False)
    _set_ball(s, [0, 0, R])
    pos, vel = s.ball.position.tolist(), [0.0, 0.0, 0.0]
    for speed in (0.0, 800.0, 1800.0):
        _place(s, "blue-0", [0, -3000, 17], yaw=np.pi / 2, vel=(0, speed, 0))
        s.cars["blue-0"].on_ground = True
        ground = car_eta(s.cars["blue-0"], pos, vel)[0]
        _place(s, "blue-0", [0, -3000, 60], yaw=np.pi / 2, vel=(0, speed, 300))       # juste après un saut
        s.cars["blue-0"].on_ground = False
        air = car_eta(s.cars["blue-0"], pos, vel)[0]
        assert ground <= air < ground + 0.8, (speed, ground, air)


def test_race_info_mirror_symmetric_and_cached():
    s = make_state(2, kickoff=True, seed=4)
    _set_ball(s, [600, -900, 400], [300, -1200, 200])
    s.cars["blue-1"].boost_amount = 0.0
    m = _mirror_state(s)
    a, b = compute_race(s), compute_race(m)
    for agent in s.cars:
        assert np.isclose(a.eta[agent], b.eta[_swap(agent)], atol=1e-4)
    assert np.isclose(a.goal_in[0], b.goal_in[1]) and np.isclose(a.goal_in[1], b.goal_in[0])
    info = {}
    assert race_info(s, info) is race_info(s, info)                  # même step : calcul partagé


# --------------------------------------------------------------------------- potentiels
def _run(term, s, agents, info, steps):
    """Applique `steps` (fonctions qui modifient l'état) et renvoie la somme des rewards par agent."""
    term.reset(agents, s, info)
    total = {a: 0.0 for a in agents}
    for f in steps:
        f(s)
        for a, v in term.get(agents, s, info).items():
            total[a] += v
    return total


def test_potential_rewards_telescope():
    """γ = 1 : la somme sur un épisode = Φ(fin) − Φ(début), quel que soit le chemin (pas de farming possible)."""
    s = make_state(1, kickoff=False)
    agents = list(s.cars)
    _set_ball(s, [0, 0, R])
    moves = [lambda st, y=y: _set_ball(st, [500, y, R]) for y in (1000, -2000, 3000, -500, 0)]
    total = _run(BallGoalPotential(gamma=1.0), s, agents, {}, moves)
    phi = BallGoalPotential(1.0).potentials(agents, s, {})
    start = make_state(1, kickoff=False)
    _set_ball(start, [0, 0, R])
    phi0 = BallGoalPotential(1.0).potentials(agents, start, {})
    for a in agents:
        assert np.isclose(total[a], phi[a] - phi0[a], atol=1e-6)
    assert np.isclose(phi["blue-0"], -phi["orange-0"])                # antisymétrique


def test_threat_rewards_shot_and_save():
    s = make_state(1, kickoff=True)
    agents = list(s.cars)
    term = ThreatPotential(gamma=1.0)
    info = {}
    _set_ball(s, [0, 2500, R], [0, 0, 0])
    term.reset(agents, s, info)
    s.tick_count += 8
    _set_ball(s, [0, 2600, R], [0, 2400, 0])                         # le bleu tire, cadré
    r = term.get(agents, s, info)
    assert r["blue-0"] > 0.5 and np.isclose(r["orange-0"], -r["blue-0"])
    s.tick_count += 8
    _set_ball(s, [0, 3000, R], [0, 2400, 0])                         # le tir approche : la menace grandit
    r = term.get(agents, s, info)
    assert r["orange-0"] < 0
    s.tick_count += 8
    _set_ball(s, [0, 3200, R], [900, -1500, 0])                      # l'orange arrête le tir
    r = term.get(agents, s, info)
    assert r["orange-0"] > 0.5 and r["blue-0"] < -0.5


def test_boost_gain_below_threshold():
    from rlagent.env.rewards import BoostGainTerm
    s = make_state(1, kickoff=False)
    t = BoostGainTerm(below=50.0)
    s.cars["blue-0"].boost_amount = 20.0
    t.reset(["blue-0"], s, {})
    s.cars["blue-0"].boost_amount = 32.0
    assert t.get(["blue-0"], s, {})["blue-0"] > 0                    # à sec : ramassage payé
    s.cars["blue-0"].boost_amount = 60.0
    t.get(["blue-0"], s, {})
    s.cars["blue-0"].boost_amount = 72.0
    assert t.get(["blue-0"], s, {})["blue-0"] == 0.0                  # déjà plein à 60 : rien


def test_s7d_boost_potential_no_leak_and_vel_cap():
    """B1 : avec boost_potential_gamma = 1, garder son boost ne coûte rien ; B2 : l'aide d'approche ne paie plus
    au-delà de la vitesse max sans boost."""
    from rlagent.env.rewards import VelPlayerToBallTerm
    s = make_state(1, kickoff=False)
    agents = list(s.cars)
    s.cars["blue-0"].boost_amount = 100.0
    leak = BoostPotential(gamma=0.99)
    tight = BoostPotential(gamma=1.0)
    leak.reset(agents, s, {}); tight.reset(agents, s, {})
    assert leak.get(agents, s, {})["blue-0"] < -0.009 and tight.get(agents, s, {})["blue-0"] == 0.0
    cfg = load_config(["s7", "s7b", "s7c", "s7d"])
    names = {t.name: t for t in make_reward(cfg).terms}
    assert names["boost_potential"].gamma == 1.0 and names["vel_player_to_ball"].cap == 1410.0
    car = s.cars["blue-0"].physics
    _set_ball(s, [0, 0, R])
    car.position = np.array([-2000, 0, 17], dtype=np.float32)
    capped, free = VelPlayerToBallTerm(cap=1410.0), VelPlayerToBallTerm()
    for vx, want_capped in ((1000.0, 1000.0), (2300.0, 1410.0)):
        car.linear_velocity = np.array([vx, 0, 0], dtype=np.float32)
        assert np.isclose(capped.get(["blue-0"], s, {})["blue-0"], want_capped / cv.CAR_MAX_SPEED, rtol=2e-3)
        assert np.isclose(free.get(["blue-0"], s, {})["blue-0"], vx / cv.CAR_MAX_SPEED, rtol=2e-3)


def test_boost_potential_pickup_and_spend():
    s = make_state(1, kickoff=False)
    agents = list(s.cars)
    t = BoostPotential(gamma=1.0)
    s.cars["blue-0"].boost_amount = 0.0
    t.reset(agents, s, {})
    s.cars["blue-0"].boost_amount = 12.0
    up = t.get(agents, s, {})["blue-0"]
    s.cars["blue-0"].boost_amount = 0.0
    down = t.get(agents, s, {})["blue-0"]
    assert np.isclose(up, np.sqrt(0.12)) and np.isclose(down, -up)


def test_positioning_attack_when_winning_defend_when_losing():
    s = make_state(1, kickoff=False)
    agents = list(s.cars)
    term = PositioningPotential(gamma=1.0)
    _set_ball(s, [0, 0, R])
    # le bleu est tout près de la balle, l'orange loin : le bleu gagne la course → Φ ≈ approche
    _place(s, "blue-0", [0, -600, 17], yaw=np.pi / 2)
    _place(s, "orange-0", [0, 4500, 17], yaw=np.pi / 2)               # dos à la balle
    phi = term.potentials(agents, s, {})
    assert phi["blue-0"] > 0.7
    # l'orange perd la course : être entre la balle et son but (en +Y) vaut plus qu'être derrière la balle
    goalside = phi["orange-0"]
    _place(s, "orange-0", [0, -2500, 17], yaw=np.pi / 2)              # du mauvais côté de la balle
    beaten = term.potentials(agents, s, {})["orange-0"]
    assert goalside > 0.5 and beaten < 0


# --------------------------------------------------------------------------- compétences
def test_aerial_touch_height_chain_cooldown_and_walls():
    s = make_state(1, kickoff=False)
    agents = list(s.cars)
    t = AerialTouchTerm()
    t.reset(agents, s, {})
    car = s.cars["blue-0"]
    _set_ball(s, [0, 0, 1720])
    car.ball_touches, car.on_ground = 1, False
    s.tick_count = 1000
    first = t.get(agents, s, {})["blue-0"]
    assert np.isclose(first, 2.0) and t.get(agents, s, {})["orange-0"] == 0.0
    s.tick_count += 8                                                 # contact prolongé : pas repayé
    assert t.get(agents, s, {})["blue-0"] == 0.0
    s.tick_count += 120                                               # 2e touche aérienne sans retomber : ×1,5
    assert np.isclose(t.get(agents, s, {})["blue-0"], 3.0)
    car.on_ground = True                                              # au mur / au sol : rien, et la chaîne repart
    s.tick_count += 120
    assert t.get(agents, s, {})["blue-0"] == 0.0
    car.on_ground = False
    _set_ball(s, [0, 0, 150])                                         # balle basse : pas une aérienne
    s.tick_count += 120
    assert t.get(agents, s, {})["blue-0"] == 0.0
    _set_ball(s, [0, 0, 300])
    s.tick_count += 120
    assert 0.5 < t.get(agents, s, {})["blue-0"] < 0.7                 # touche en sautant


def test_kickoff_terms():
    s = make_state(2, kickoff=True, seed=1)
    s.tick_count = 0
    agents = list(s.cars)
    info = {"start_kind": "kickoff"}
    win, rush = KickoffWinTerm(), KickoffRushTerm()
    win.reset(agents, s, info)
    rush.reset(agents, s, info)
    for c in s.cars.values():                                         # tout le monde fonce vers la balle
        d = -c.physics.position / np.linalg.norm(c.physics.position)
        c.physics.linear_velocity = (d * 1500).astype(np.float32)
    r = rush.get(agents, s, info)
    for team in ("blue", "orange"):
        mates = [a for a in agents if a.startswith(team)]
        assert sum(r[a] > 0.5 for a in mates) == 1                    # seul le 1er homme est poussé
    s.tick_count = int(3.1 * 120)
    _set_ball(s, [0, 1500, R])                                        # balle dans le camp orange
    w = win.get(agents, s, info)
    assert np.isclose(w["blue-0"], 0.75) and np.isclose(w["orange-1"], -0.75)
    assert all(v == 0.0 for v in win.get(agents, s, info).values())   # versé une seule fois
    random_start = KickoffWinTerm()
    random_start.reset(agents, s, {"start_kind": "random"})
    assert all(v == 0.0 for v in random_start.get(agents, s, {}).values())


def test_concede_only_hurts_conceding_team():
    s = make_state(2, kickoff=False)
    s.goal_scored = True
    _set_ball(s, [0, -5300, 200])                                     # dans le but bleu : l'orange marque
    r = ConcedeTerm().get(list(s.cars), s, {})
    assert r["blue-0"] == r["blue-1"] == -1.0 and r["orange-0"] == r["orange-1"] == 0.0


def test_s7_reward_symmetric_and_finite():
    cfg = load_config(["s7"])
    rf_a, rf_b = make_reward(cfg), make_reward(cfg)
    s = make_state(2, kickoff=True, seed=2)
    _set_ball(s, [400, -700, 500], [300, -900, 250])
    s.cars["blue-0"].boost_amount = 80.0
    m = _mirror_state(s)
    ia, ib = {"global_step": 0, "start_kind": "random"}, {"global_step": 0, "start_kind": "random"}
    rf_a.reset(list(s.cars), s, ia)
    rf_b.reset(list(m.cars), m, ib)
    for _ in range(3):
        for st in (s, m):
            st.tick_count += 8
        s.ball.position = s.ball.position + np.array([40, -120, 10], dtype=np.float32)
        m.ball.position = m.ball.position + np.array([-40, 120, 10], dtype=np.float32)
        ra = rf_a.get_rewards(list(s.cars), s, {}, {}, ia)
        rb = rf_b.get_rewards(list(m.cars), m, {}, {}, ib)
        for a in s.cars:
            assert np.isfinite(ra[a]) and np.isclose(ra[a], rb[_swap(a)], atol=1e-4)
    assert {"threat", "positioning", "aerial_touch", "boost_potential"} <= set(ia["reward_log"])


def test_anneal_begin_offsets_schedule():
    from rlagent.env.rewards import TeamReward, TouchTerm
    tr = TeamReward([TouchTerm(0.0)], {"touch": 1.0}, anneal={"touch": {"begin": 100, "end": 0.0, "steps": 100}})
    assert tr.current_weight("touch", 50) == 1.0
    assert np.isclose(tr.current_weight("touch", 150), 0.5)
    assert tr.current_weight("touch", 400) == 0.0


# --------------------------------------------------------------------------- observations
def test_race_features_dims_prefix_and_symmetry():
    from rlagent.env.obs import BALL_PRED_DIM, RACE_DIM, SELF_DIM, AttentionObs, obs_dim
    s = make_state(3, kickoff=True, seed=6)
    _set_ball(s, [-300, 800, 700], [-200, 900, 300])
    bp = AttentionObs(True).build_obs(list(s.cars), s, {})
    race = AttentionObs(True, True).build_obs(list(s.cars), s, {})
    k = SELF_DIM + BALL_PRED_DIM
    for a in s.cars:
        assert race[a].shape == (obs_dim(True, True),) == (obs_dim(True) + RACE_DIM,)
        assert np.array_equal(np.concatenate([race[a][:k], race[a][k + RACE_DIM:]]), bp[a])
        assert np.all(np.abs(race[a][k:k + RACE_DIM]) <= 1.0 + 1e-6)
    m = _mirror_state(s)
    race_m = AttentionObs(True, True).build_obs(list(m.cars), m, {})
    for a in s.cars:
        assert np.allclose(race[a], race_m[_swap(a)], atol=1e-4)
    with pytest.raises(ValueError):
        AttentionObs(False, True)


torch = pytest.importorskip("torch")


def test_older_networks_play_on_race_obs():
    from rlagent.env.obs import AttentionObs
    from rlagent.models.attention import build_models
    s = make_state(2, kickoff=True, seed=7)
    _set_ball(s, [100, 200, 300], [400, -300, 100])
    obs = {k: np.stack(list(AttentionObs(*k).build_obs(list(s.cars), s, {}).values()))
           for k in ((False, False), (True, False), (True, True))}
    for bp in (False, True):
        cfg = load_config(["debug"], [f"env.ball_prediction={str(bp).lower()}"])
        torch.manual_seed(0)
        actor, _ = build_models(cfg, 90)
        with torch.no_grad():
            ref = actor(torch.as_tensor(obs[(bp, False)]))
            assert torch.allclose(actor(torch.as_tensor(obs[(True, True)])), ref, atol=1e-6)


# --------------------------------------------------------------------------- exercices à niveaux
def test_aerial_drill_levels_and_success_threshold():
    from rlagent.env.mutators import DRILLS
    from rlagent.metrics.episode import EpisodeTracker
    # rendez-vous (s7d) : la balle passe au-dessus de M à la hauteur h à l'instant T où la voiture y arrive
    for level, (lo, hi) in ((0.0, (200, 280)), (1.0, (1150, 1800))):
        for seed in range(15):
            s = make_state(1, kickoff=True, seed=seed)
            info = {"drill_levels": {"aerial": level}}
            drill = DRILLS["aerial"](seed)
            drill.apply(s, info)
            mx, my, h, T = drill.last_meet
            assert info["drill_level"] == level and lo <= h <= hi and 0.3 < T < 4.0
            team = info["drill_team"]
            car = next(c for c in s.cars.values() if c.team_num == team).physics
            to_m = np.array([mx, my]) - car.position[:2]
            speed = float(np.linalg.norm(car.linear_velocity[:2]))
            cos_min = 0.95 if level == 0.0 else 0.3                         # roule vers M (bruit de cap au niveau 1)
            assert np.dot(car.linear_velocity[:2], to_m) > cos_min * speed * np.linalg.norm(to_m)
            b, v = s.ball.position, s.ball.linear_velocity   # chute libre sans frottement : ~ (M, h), à la
            tol = 0.02 * T * T                                # compensation du frottement près (×(1 + 0,0305·T/2))
            assert np.allclose(b[:2] + v[:2] * T, [mx, my], atol=tol * np.linalg.norm(v[:2]) + 1)
            assert abs(b[2] + v[2] * T - 325.0 * T * T - h) < tol * abs(v[2]) + 1
            vz0 = max(v[2], 0) / (1 + 0.0305 * T / 2)                      # sans la compensation du frottement
            assert b[2] >= 149 and b[2] + vz0 ** 2 / 1300 <= 1900           # pas dans le sol ni au plafond
    s = make_state(1, kickoff=True)
    info = {}
    DRILLS["aerial"](0).apply(s, info)
    assert "drill_level" not in info and s.ball.position[2] >= 600     # sans curriculum : réglage s6
    # niveau 0 : une touche en sautant (balle à 300 uu) réussit ; sans niveau il faut 400 uu
    for level, ok in ((0.0, 1.0), (None, 0.0)):
        s = make_state(1, kickoff=True)
        s.tick_count = 0
        t = EpisodeTracker()
        t.reset(s, {"start_kind": "drill_aerial", "drill_team": 0, "team_size": 1,
                    **({} if level is None else {"drill_level": level})})
        s.tick_count = 120
        _set_ball(s, [0, 0, 300])
        s.cars["blue-0"].ball_touches = 1
        t.step(s)
        out = t.finish(s, False)
        assert out["drill_success"] == ok
        assert ("drill_level" in out) == (level is not None)


def test_aerial_defense_lob_heads_into_own_goal():
    from rlagent.env.mutators import DRILLS
    drill = DRILLS["aerial_defense"](0)
    for k in range(300):                                              # beaucoup de tirages : aucun NaN, toujours cadré
        s = make_state(1, kickoff=True, seed=k)
        info = {"drill_levels": {"aerial_defense": (k % 11) / 10}}
        drill.apply(s, info)
        assert np.all(np.isfinite(s.ball.linear_velocity)), k
        sign = -1.0 if info["drill_team"] == cv.BLUE_TEAM else 1.0
        t = goal_entry_time(s.ball.position.tolist(), s.ball.linear_velocity.tolist(), sign, horizon=6.0)
        assert t is not None, k


def test_curriculum_promotes_demotes_and_roundtrips():
    from rlagent.train import DrillCurriculum
    c = DrillCurriculum(["aerial", "defense"], step=0.1, ema=0.5)
    for _ in range(5):
        c.update([{"drill": "aerial", "drill_success": 1.0, "drill_level": 0.0},
                  {"drill": "defense", "drill_success": 0.0, "drill_level": 0.0}])
    assert c.levels["aerial"] > 0.3 and c.levels["defense"] == 0.0
    c.update([{"drill": "shot", "drill_success": 1.0}])               # exercice hors curriculum : ignoré
    d = DrillCurriculum(["aerial", "defense"])
    d.load_state_dict(c.state_dict())
    assert d.levels == c.levels
    cfg = load_config(["s7"])
    assert set(DrillCurriculum.from_cfg(cfg).levels) == {"aerial", "aerial_defense", "defense", "shot"}
    assert DrillCurriculum.from_cfg(load_config()) is None


def test_batch_schedule():
    from rlagent.train import scheduled_batch
    cfg = load_config(["s7"])
    assert scheduled_batch(cfg, 0) == (50_000, 25_000)
    assert scheduled_batch(cfg, 30_000_000) == (100_000, 50_000)
    assert scheduled_batch(cfg, 10**9) == (200_000, 50_000)
    assert scheduled_batch(load_config(), 10**9) == (100_000, 50_000)  # sans liste : valeurs fixes


def test_tracker_counts_shots_and_saves():
    from rlagent.metrics.episode import EpisodeTracker
    s = make_state(1, kickoff=True)
    s.tick_count = 0
    _set_ball(s, [0, 2500, R])
    t = EpisodeTracker()
    t.reset(s, {"start_kind": "random", "team_size": 1})
    s.tick_count = 8
    _set_ball(s, [0, 2600, R], [0, 2500, 0])                         # le bleu tire cadré
    s.cars["blue-0"].ball_touches = 1
    t.step(s)
    s.tick_count = 16
    s.cars["blue-0"].ball_touches = 0
    s.cars["orange-0"].ball_touches = 1
    _set_ball(s, [0, 3000, R], [1200, -1000, 0])                     # l'orange arrête
    t.step(s)
    out = t.finish(s, False)
    assert out["shots_per_min"] > 0 and out["saves_per_min"] > 0
    assert {"flips_per_min", "low_boost_frac", "avg_speed"} <= set(out["mech"])


def test_training_s7_profile_on_fake_engine(tmp_path):
    """Chaîne complète avec le profil s7 : nouvelles observations, rewards, curriculum (envoyé aux workers et
    sauvegardé), lot progressif ; puis reprise depuis le checkpoint."""
    import json
    from rlagent.train import find_latest_checkpoint, train
    cfg = load_config(["debug", "s7"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["format_weights"] = {1: 0.5, 2: 0.3, 3: 0.2}
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    cfg["env"]["drill_curriculum"].update({"step": 0.2, "ema": 0.5})
    cfg["ppo"]["batch_schedule"] = [[0, 2000, 1000], [4000, 3000, 1500]]
    cfg["ppo"]["device"] = "cpu"
    cfg["train"].update({"n_workers": 2, "total_steps": 9000, "checkpoint_every_steps": 3000})
    cfg["selfplay"].update({"pool_every_steps": 3000, "hist_prob": 0.5})
    cfg["run_name"] = "pytest_s7"
    train(cfg, tmp_path)
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(l["learn/policy_loss"]) and np.isfinite(l["learn/value_loss"]) for l in lines)
    assert lines[0]["learn/batch_steps"] < 3000 <= lines[-1]["learn/batch_steps"]
    keys = set().union(*lines)
    assert {"curriculum/aerial/level", "reward_term/threat", "reward_term/positioning", "shots_per_min"} <= keys
    ck = torch.load(find_latest_checkpoint(tmp_path), map_location="cpu", weights_only=False)
    assert set(ck["curriculum"]["levels"]) == {"aerial", "aerial_defense", "defense", "shot"}
    assert ck["actor"]["encoder.self_embed.0.weight"].shape[1] == cfg["model"]["self_dim"]
    cfg["train"]["total_steps"] = 12000
    train(cfg, tmp_path, find_latest_checkpoint(tmp_path))
    lines2 = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines2[-1]["global_step"] >= 12000
