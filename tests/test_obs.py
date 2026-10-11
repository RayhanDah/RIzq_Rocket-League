"""S1-09 : observations normalisées côté équipe (situation miroir → obs identiques), taille fixe, masque."""
import numpy as np

from helpers import make_state, mirror_car_into
from rlagent.env.obs import MAX_OTHERS, OBS_DIM, OTHER_DIM, SELF_DIM, AttentionObs


def _obs(state, prev=None):
    builder = AttentionObs()
    agents = list(state.cars.keys())
    return builder.build_obs(agents, state, {"prev_actions": prev or {}})


def test_obs_dim_all_formats():
    for size in (1, 2, 3):
        obs = _obs(make_state(size))
        assert len(obs) == 2 * size
        for o in obs.values():
            assert o.shape == (OBS_DIM,)
            assert np.all(np.isfinite(o))
            mask = o[-MAX_OTHERS:]
            assert mask.sum() == 2 * size - 1


def test_kickoff_is_mirror_symmetric():
    for size in (1, 2, 3):
        for seed in range(3):
            obs = _obs(make_state(size, seed=seed))
            for i in range(size):
                np.testing.assert_allclose(obs[f"blue-{i}"], obs[f"orange-{i}"], atol=1e-5)


def test_random_position_mirror_symmetric():
    rng = np.random.default_rng(1)
    s = make_state(1, kickoff=False)
    blue = s.cars["blue-0"]
    blue.physics.position = np.array([1234.0, -2100.0, 17.0], dtype=np.float32)
    blue.physics.linear_velocity = np.array([300.0, 900.0, 0.0], dtype=np.float32)
    blue.physics.angular_velocity = np.array([0.1, -0.2, 0.7], dtype=np.float32)
    blue.physics.euler_angles = np.array([0.1, 0.4, -0.05], dtype=np.float32)
    blue.boost_amount = 57.0
    mirror_car_into(s, "blue-0", s, "orange-0")
    s.ball.position = np.array([0.0, 0.0, 300.0], dtype=np.float32)  # balle sur l'axe : symétrique
    s.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    s.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    s.boost_pad_timers = rng.uniform(0, 10, 34).astype(np.float32)
    s.boost_pad_timers = (s.boost_pad_timers + s.boost_pad_timers[::-1]) / 2  # pads symétriques
    obs = _obs(s)
    np.testing.assert_allclose(obs["blue-0"], obs["orange-0"], atol=1e-4)


def test_teammates_listed_before_opponents():
    obs = _obs(make_state(3))
    o = obs["blue-0"]
    others = o[SELF_DIM:SELF_DIM + MAX_OTHERS * OTHER_DIM].reshape(MAX_OTHERS, OTHER_DIM)
    roles = others[:, :2]
    np.testing.assert_array_equal(roles[:2], [[1, 0], [1, 0]])
    np.testing.assert_array_equal(roles[2:], [[0, 1]] * 3)


def test_prev_action_in_obs():
    s = make_state(1)
    prev = {"blue-0": np.array([1, -1, 0, -1, 0, 1, 1, 0], dtype=np.float32)}
    with_prev = _obs(s, prev)["blue-0"]
    without = _obs(s)["blue-0"]
    assert not np.allclose(with_prev, without)


def _random_state(size: int, seed: int):
    """État quelconque (positions, vitesses, rotations, scalaires, pads) pour comparer deux implémentations."""
    from rlagent.env.mutators import RandomStateMutator
    rng = np.random.default_rng(seed)
    s = make_state(size, kickoff=False, seed=seed)
    RandomStateMutator(seed).apply(s, {})
    for c in s.cars.values():
        c.physics.euler_angles = rng.uniform(-np.pi, np.pi, 3).astype(np.float32)
        c.physics.position = (c.physics.position + rng.normal(0, 200, 3)).astype(np.float32)
        c.physics.angular_velocity = rng.normal(0, 2, 3).astype(np.float32)
        c.on_ground = bool(rng.random() < 0.5)
        c.has_jumped = bool(rng.random() < 0.5)
        c.is_holding_jump = bool(rng.random() < 0.5)
        c.air_time_since_jump = float(rng.uniform(0, 2))
        c.jump_time = float(rng.uniform(0, 0.3))
        c.demo_respawn_timer = float(rng.choice([0.0, rng.uniform(0, 3)]))
        c.boost_amount = float(rng.uniform(0, 100))
    s.boost_pad_timers = rng.uniform(0, 10, 34).astype(np.float32)
    return s


def test_vectorized_obs_matches_reference():
    """L'implémentation rapide doit donner les MÊMES observations que l'implémentation d'origine."""
    builder = AttentionObs()
    rng = np.random.default_rng(0)
    for size in (1, 2, 3):
        for seed in range(15):
            s = make_state(size, seed=seed) if seed % 3 == 0 else _random_state(size, seed)
            agents = list(s.cars.keys())
            info = {"prev_actions": {a: rng.uniform(-1, 1, 8).astype(np.float32) for a in agents[::2]}}
            fast = builder.build_obs(agents, s, info)
            ref = builder.build_obs_reference(agents, s, info)
            assert fast.keys() == ref.keys()
            for a in agents:
                assert fast[a].dtype == np.float32 and fast[a].shape == (OBS_DIM,)
                np.testing.assert_allclose(fast[a], ref[a], rtol=1e-5, atol=1e-6, err_msg=f"{size}v{size} {a}")
            # sous-ensemble d'agents (cas des tests et de RLBot)
            sub = builder.build_obs(agents[:1], s, info)
            np.testing.assert_allclose(sub[agents[0]], ref[agents[0]], rtol=1e-5, atol=1e-6)
