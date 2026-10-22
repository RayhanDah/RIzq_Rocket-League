"""s7e : pads de boost dans les observations, reward `flip_speed`, défense aérienne en rendez-vous, profil s7e,
ancres figées dans le zip d'init pour Kaggle."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from rlgym.rocket_league import common_values as cv

from helpers import make_state, mirror_car_into
from rlagent.config import load_config
from rlagent.env.obs import (BALL_PRED_DIM, BOOST_DIM, RACE_DIM, SELF_DIM, AttentionObs, boost_pad_features,
                             obs_dim, self_dim)
from rlagent.env.rewards import FlipSpeedTerm, make_reward

PADS = np.asarray(cv.BOOST_LOCATIONS, dtype=np.float64)
BIG = np.where(PADS[:, 2] > 71)[0]


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


def _swap(a):
    return a.replace("blue", "tmp").replace("orange", "blue").replace("tmp", "orange")


def _mirror_state(s):
    m = make_state(len(s.cars) // 2, kickoff=True)
    m.tick_count = s.tick_count
    m.ball.position = s.ball.position * np.array([-1, -1, 1], dtype=np.float32)
    m.ball.linear_velocity = s.ball.linear_velocity * np.array([-1, -1, 1], dtype=np.float32)
    m.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    m.boost_pad_timers = s.boost_pad_timers[::-1].copy()
    for a in s.cars:
        mirror_car_into(s, a, m, _swap(a))
    return m


# --------------------------------------------------------------------------- observations : pads de boost
def test_boost_features_dims_prefix_and_mirror_symmetry():
    s = make_state(3, kickoff=True, seed=4)
    _set_ball(s, [-300, 800, 700], [-200, 900, 300])
    s.boost_pad_timers = np.zeros(34, dtype=np.float32)
    s.boost_pad_timers[[3, 15, 30]] = [7.0, 2.5, 9.0]
    race = AttentionObs(True, True).build_obs(list(s.cars), s, {})
    full = AttentionObs(True, True, True).build_obs(list(s.cars), s, {})
    k = SELF_DIM + BALL_PRED_DIM + RACE_DIM
    assert self_dim(True, True, True) == k + BOOST_DIM and obs_dim(True, True, True) == obs_dim(True, True) + BOOST_DIM
    for a in s.cars:
        assert full[a].shape == (obs_dim(True, True, True),)
        assert np.array_equal(np.concatenate([full[a][:k], full[a][k + BOOST_DIM:]]), race[a])   # préfixe
        assert np.all(np.isfinite(full[a][k:k + BOOST_DIM]))
    m = _mirror_state(s)
    full_m = AttentionObs(True, True, True).build_obs(list(m.cars), m, {})
    for a in s.cars:
        assert np.allclose(full[a], full_m[_swap(a)], atol=1e-4)
    with pytest.raises(ValueError):
        AttentionObs(True, False, True)


def test_boost_features_pick_reachable_pads_in_car_frame():
    timers = np.zeros(34)
    pos = PADS[BIG[0]] + np.array([300.0, 0.0, -50.0])          # à 300 uu du gros pad n° 0
    fwd, right = np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    f = boost_pad_features(pos, fwd, right, timers)
    assert len(f) == BOOST_DIM
    assert f[0] == pytest.approx(-300 / 2300, abs=1e-3) and f[1] == pytest.approx(0.0, abs=0.03)   # derrière soi
    assert f[2] == pytest.approx(300 / 1400 / 6, abs=1e-3) and f[3] == 0.0
    assert f[4 + 2] >= f[2]                                       # 2e gros pad : plus coûteux
    assert all(abs(f[8 + i]) <= 4.5 for i in range(2)) and f[8 + 3] == 0.0  # le petit pad
    timers[BIG[0]] = 8.0                                          # pris : 8 s avant réapparition → il passe après
    g = boost_pad_features(pos, fwd, right, timers)
    assert g[0] != pytest.approx(f[0]) and g[3] == 0.0           # le 1er emplacement est un autre gros pad
    assert g[4 + 3] == pytest.approx(0.8) or g[2] <= 8.0 / 6      # ... ou ce pad redevient le 2e choix


def test_older_networks_play_on_boost_obs_and_expansion_is_exact():
    torch = pytest.importorskip("torch")
    from rlagent.models.attention import build_models
    from rlagent.train import load_state_expanding
    s = make_state(2, kickoff=True, seed=8)
    _set_ball(s, [100, 200, 300], [400, -300, 100])
    race = np.stack(list(AttentionObs(True, True).build_obs(list(s.cars), s, {}).values()))
    full = np.stack(list(AttentionObs(True, True, True).build_obs(list(s.cars), s, {}).values()))
    cfg_old = load_config(["debug"], ["env.ball_prediction=true", "env.race_features=true"])
    cfg_new = load_config(["debug"], ["env.ball_prediction=true", "env.race_features=true", "env.boost_features=true"])
    assert cfg_new["model"]["self_dim"] == cfg_old["model"]["self_dim"] + BOOST_DIM
    torch.manual_seed(0)
    old_a, old_c = build_models(cfg_old, 90)
    new_a, new_c = build_models(cfg_new, 90)
    with torch.no_grad():
        ref = old_a(torch.as_tensor(race))
        assert torch.allclose(old_a(torch.as_tensor(full)), ref, atol=1e-6)          # fit_obs
        assert load_state_expanding(new_a, old_a.state_dict()) == ["encoder.self_embed.0.weight"]
        load_state_expanding(new_c, old_c.state_dict())
        assert torch.allclose(new_a(torch.as_tensor(full)), ref, atol=1e-6)
        assert torch.allclose(new_c(torch.as_tensor(full)), old_c(torch.as_tensor(race)), atol=1e-6)


# --------------------------------------------------------------------------- reward flip_speed
def _flip_run(speeds, flip_at, boosting=(), min_speed=1000.0):
    """Voiture qui roule vers la balle (+X) aux vitesses données (1 step = 8 ticks) ; flip au step `flip_at`."""
    s = make_state(1, kickoff=True)
    s.tick_count = 0
    _set_ball(s, [3000, 0, 93])
    _place(s, "blue-0", [-2000, 0, 17], vel=(speeds[0], 0, 0))
    s.cars["blue-0"].has_flipped = False
    term = FlipSpeedTerm(min_speed=min_speed)
    term.reset(["blue-0"], s, {})
    out = []
    for i, v in enumerate(speeds):
        s.tick_count = 8 * (i + 1)
        car = s.cars["blue-0"]
        car.physics.linear_velocity = np.array([v, 0, 0], dtype=np.float32)
        car.has_flipped = i >= flip_at
        car.boost_active_time = 0.1 if i in boosting else 0.0
        out.append(term.get(["blue-0"], s, {})["blue-0"])
    return out


def test_flip_speed_pays_flip_gain_only():
    # flip à 1000 uu/s qui mène à 1500 : +500 → 1 ; versé une fois, ~0,3 s après le début du flip
    r = _flip_run([1000] * 4 + [1100, 1300, 1500, 1500, 1500, 1500], flip_at=4)
    assert sum(r) == pytest.approx(1.0) and r.index(1.0) == 4 + 5
    # freiner puis flipper : le record de la dernière seconde (1400) sert de référence → seulement +100
    r = _flip_run([1400] * 4 + [900, 900, 1100, 1300, 1500, 1500, 1500, 1500], flip_at=6)
    assert sum(r) == pytest.approx(100 / 500)
    # pas de flip, flip qui ralentit, flip trop lent, boost pendant le flip (déduit)
    assert sum(_flip_run([1000, 1200, 1500, 1700, 1900, 2000, 2000], flip_at=99)) == 0.0
    assert sum(_flip_run([1500] * 4 + [1400, 1300, 1200, 1200, 1200, 1200], flip_at=4)) == 0.0
    assert sum(_flip_run([300] * 4 + [500, 700, 800, 800, 800, 800], flip_at=4)) == 0.0
    r = _flip_run([1000] * 4 + [1150, 1400, 1700, 1700, 1700, 1700], flip_at=4, boosting={4, 5, 6})
    boost_gain = FlipSpeedTerm.BOOST_ACCEL * 24 / 120          # 3 steps de boost = 24 ticks
    assert sum(r) == pytest.approx(min(1.0, (700 - boost_gain) / 500), abs=1e-3)


def test_s7e_reward_contains_new_terms_and_stays_symmetric():
    cfg = load_config(["s7", "s7b", "s7c", "s7d", "s7e"])
    rf = make_reward(cfg)
    names = {t.name for t in rf.terms}
    assert {"flip_speed", "shot", "aerial_touch", "boost_gain"} <= names
    assert rf.current_weight("boost_gain", 10 ** 9) == pytest.approx(0.4)          # plus de décroissance
    assert rf.current_weight("flip_speed", 72_000_000) == pytest.approx(0.25)
    assert rf.current_weight("flip_speed", 222_000_000) == pytest.approx(0.1)
    rf_m = make_reward(cfg)
    s = make_state(2, kickoff=True, seed=2)
    _set_ball(s, [400, -700, 500], [300, -900, 250])
    m = _mirror_state(s)
    ia, ib = {"global_step": 80_000_000, "start_kind": "random"}, {"global_step": 80_000_000, "start_kind": "random"}
    rf.reset(list(s.cars), s, ia)
    rf_m.reset(list(m.cars), m, ib)
    for _ in range(3):
        for st in (s, m):
            st.tick_count += 8
        s.ball.position = s.ball.position + np.array([40, -120, 10], dtype=np.float32)
        m.ball.position = m.ball.position + np.array([-40, 120, 10], dtype=np.float32)
        ra = rf.get_rewards(list(s.cars), s, {}, {}, ia)
        rb = rf_m.get_rewards(list(m.cars), m, {}, {}, ib)
        for a in s.cars:
            assert np.isfinite(ra[a]) and np.isclose(ra[a], rb[_swap(a)], atol=1e-4)


# --------------------------------------------------------------------------- défense aérienne en rendez-vous
def test_aerial_defense_rendezvous_geometry():
    from rlagent.env.mutators import DRILLS
    from rlagent.env.predict import goal_entry_time
    drill = DRILLS["aerial_defense"](0)
    g = drill.GRAVITY
    for level, (lo, hi) in ((0.0, (140, 260)), (0.5, (320, 680)), (1.0, (500, 1100))):
        for seed in range(40):
            s = make_state(1, kickoff=True, seed=seed)
            info = {"drill_levels": {"aerial_defense": level}}
            drill.apply(s, info)
            mx, my, h, T = drill.last_meet
            team = info["drill_team"]
            own_y = (-1 if team == cv.BLUE_TEAM else 1) * cv.BACK_WALL_Y
            assert lo <= h <= hi and 0.9 < T < 1.9
            b, v = s.ball.position.astype(float), s.ball.linear_velocity.astype(float)
            drag = 1 + 0.0305 * T / 2
            assert np.allclose(b[:2] + v[:2] / drag * T, [mx, my], atol=1.0)            # au-dessus de M à T
            assert abs(b[2] + v[2] / drag * T - 0.5 * g * T * T - h) < 1.0
            assert b[2] >= 119 and b[2] + max(v[2] / drag, 0) ** 2 / (2 * g) < 1400    # ni sol ni plafond
            sign = -1.0 if team == cv.BLUE_TEAM else 1.0
            assert goal_entry_time(b.tolist(), v.tolist(), sign, horizon=6.0) is not None, (level, seed)
            car = next(c for c in s.cars.values() if c.team_num == team).physics
            assert abs(car.position[1]) <= abs(own_y) - 149                          # devant la ligne
            gap = np.array([mx, my]) - car.position[:2]
            assert 200 < np.linalg.norm(gap) < 2400                                  # près de M, pas dessus
    s = make_state(1, kickoff=True)
    info = {}
    drill.apply(s, info)                                                            # sans curriculum : ancien réglage
    assert "drill_level" not in info and np.isfinite(s.ball.linear_velocity).all()


# --------------------------------------------------------------------------- profil et bundle Kaggle
def test_s7e_profile():
    from rlagent.train import scheduled_batch
    cfg = load_config(["s7", "s7b", "s7c", "s7d", "s7e"])
    assert cfg["env"]["boost_features"] and cfg["model"]["self_dim"] == self_dim(True, True, True)
    assert cfg["ppo"]["gamma"] == 0.995 and cfg["ppo"]["lr_max"] == pytest.approx(3e-3)
    assert scheduled_batch(cfg, 10 ** 9) == (100_000, 50_000)
    assert cfg["selfplay"]["pfsp"] and len(cfg["selfplay"]["anchors"]) == 4
    assert sum(cfg["env"]["start_weights"].values()) == pytest.approx(1.0)
    assert math.isclose(cfg["rewards"]["boost_potential_gamma"], 1.0)                 # s7d conservé


def test_bundle_freezes_anchors_reused_without_sources(tmp_path):
    torch = pytest.importorskip("torch")
    from rlagent.env.actions import N_ACTIONS
    from rlagent.models.attention import Actor
    from rlagent.selfplay.pool import OpponentPool
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_cloud_bundle import freeze_anchors
    run = tmp_path / "src_runs" / "s6_fake"
    torch.manual_seed(3)
    actor = Actor(N_ACTIONS, d_model=16, n_heads=2, n_layers=1, ff_mult=2, head_hidden=(16,), self_dim=SELF_DIM)
    (run / "checkpoints").mkdir(parents=True)
    model = {"d_model": 16, "n_heads": 2, "n_layers": 1, "ff_mult": 2, "head_hidden": [16]}
    torch.save({"actor": actor.state_dict(), "cfg": {"model": model}, "global_step": 5, "elo": 1500.0},
               run / "checkpoints" / "ckpt_0000000000005.pt")
    spec = run.as_posix()
    prof = tmp_path / "anchors.yaml"
    prof.write_text(f"selfplay:\n  anchors: ['{spec}']\n", encoding="utf-8")
    files = freeze_anchors([str(prof)], tmp_path / "frozen")
    assert [f.name for f in files] == ["s6_fake.pt"]
    # sur « Kaggle » : la copie est dans pool/anchors/, la source n'existe pas
    kaggle_pool = tmp_path / "kaggle" / "pool"
    (kaggle_pool / "anchors").mkdir(parents=True)
    (kaggle_pool / "anchors" / "s6_fake.pt").write_bytes(files[0].read_bytes())
    import shutil
    shutil.rmtree(run)
    pool = OpponentPool(kaggle_pool, anchors=[spec], anchor_root=tmp_path / "nowhere")
    assert [a["name"] for a in pool.anchors] == ["s6_fake"] and pool.anchors[0]["step"] == 5


def test_s7f_profile_only_changes_batch_and_entropy():
    from rlagent.train import scheduled_batch
    e = load_config(["s7", "s7b", "s7c", "s7d", "s7e"])
    f = load_config(["s7", "s7b", "s7c", "s7d", "s7e", "s7f"])
    assert scheduled_batch(f, 105_000_000) == (1_000_000, 500_000) == scheduled_batch(f, 10 ** 9)
    assert f["ppo"]["ent_coef"] == pytest.approx(0.005) and f["train"]["init_from"] == "runs/s7e_spo_laptop"
    for section in ("env", "rewards", "selfplay", "model"):                  # infrastructure de s7e inchangée
        assert f[section] == e[section], section
    diff = {k for k in f["ppo"] if f["ppo"][k] != e["ppo"][k]}
    assert diff == {"ent_coef", "batch_schedule"}
