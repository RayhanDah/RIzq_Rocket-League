"""Refonte s10 (23/10/2026) : acteur / critique MLP, rewards GigaLearn (frappe forte, démo, bump, vitesse signée),
poids par palier, curriculum par paliers de compétence, échelle Elo absolue, et le profil complet sur moteur factice."""
import json

import numpy as np
import pytest

from helpers import make_state
from rlagent.config import load_config
from rlagent.curriculum import SkillCurriculum
from rlagent.env.rewards import BumpTerm, DemoTerm, StrongTouchTerm, VelPlayerToBallTerm, make_reward
from rlagent.selfplay.ladder import CURRENT, Ladder


# --------------------------------------------------------------------------- profil
@pytest.mark.parametrize("infra", [[], ["laptop", "laptop_xpu"], ["kaggle"], ["kaggle_tpu"]])
def test_s10_effective_config(infra):
    """Config RÉELLE après fusion avec default.yaml et les profils d'infra (les dictionnaires fusionnent : une clé
    oubliée garde la valeur par défaut, ex. les formats 2v2/3v3 du 1er run de validation)."""
    cfg = load_config(infra + ["s10"])
    e, r = cfg["env"], cfg["rewards"]
    assert {k: v for k, v in e["format_weights"].items() if v > 0} == {1: 1.0}
    assert {k: v for k, v in e["start_weights"].items() if v > 0} == {"kickoff": 0.5, "random": 0.5}
    assert not e.get("adaptive_drills") and (e["ball_prediction"], e["race_features"], e["boost_features"]) == (
        True, False, False)
    assert r["anneal"] is None and r["team_spirit"]["end"] == 0.0
    assert r["weights"]["double_commit"] == 0.0 and r["weights"]["vel_player_to_ball"] == 0.27
    assert r["weights"]["flip_speed"] == 0.5 and r["flip_speed_min_speed"] == 300.0
    p = cfg["ppo"]
    assert (p["policy_loss"], p["lr_schedule"], p["batch_steps"], p["minibatch_size"], p["epochs"]) == (
        "ppo", "fixed", 100_000, 50_000, 2)
    assert cfg["model"]["type"] == "mlp" and cfg["train"]["init_from"] is None and cfg["selfplay"]["ladder"]


# --------------------------------------------------------------------------- rewards
def test_strong_touch_threshold_and_cap():
    s = make_state(1)
    t = StrongTouchTerm()
    t.reset(list(s.cars), s, {})
    s.cars["blue-0"].ball_touches = 1
    s.ball.linear_velocity = np.array([0, 400.0, 0], dtype=np.float32)       # < 20 km/h : simple contact
    assert t.get(list(s.cars), s, {}) == {"blue-0": 0.0, "orange-0": 0.0}
    s.ball.linear_velocity = np.array([0, 400.0 + 1389.0, 0], dtype=np.float32)  # Δv = 50 km/h
    r = t.get(list(s.cars), s, {})
    assert np.isclose(r["blue-0"], 0.5, atol=1e-3) and r["orange-0"] == 0.0
    s.ball.linear_velocity = np.array([0, -4000.0, 0], dtype=np.float32)     # très forte : plafonnée
    assert t.get(list(s.cars), s, {})["blue-0"] == 1.0


def test_vel_player_to_ball_signed_option():
    s = make_state(1)
    car = s.cars["blue-0"]
    to_ball = s.ball.position - car.physics.position
    car.physics.linear_velocity = (-to_ball / np.linalg.norm(to_ball) * 1150).astype(np.float32)
    assert VelPlayerToBallTerm().get(["blue-0"], s, {})["blue-0"] == 0.0
    assert np.isclose(VelPlayerToBallTerm(signed=True).get(["blue-0"], s, {})["blue-0"], -0.5, atol=1e-4)


def test_demo_rewards_author_once():
    s = make_state(1)
    t = DemoTerm()
    agents = list(s.cars)
    t.reset(agents, s, {})
    s.cars["orange-0"].demo_respawn_timer = 3.0
    s.cars["blue-0"].bump_victim_id = "orange-0"
    assert t.get(agents, s, {}) == {"blue-0": 1.0, "orange-0": 0.0}
    assert t.get(agents, s, {}) == {"blue-0": 0.0, "orange-0": 0.0}        # toujours démoli : pas de 2e paiement
    # sans contact enregistré : l'adversaire le plus proche est l'auteur
    s2 = make_state(2)
    t.reset(list(s2.cars), s2, {})
    s2.cars["blue-1"].demo_respawn_timer = 3.0
    s2.cars["orange-0"].physics.position = s2.cars["blue-1"].physics.position + np.float32(10.0)
    r = t.get(list(s2.cars), s2, {})
    assert r["orange-0"] == 1.0 and sum(r.values()) == 1.0


def test_bump_paid_once_per_contact_and_only_on_opponents():
    s = make_state(2)
    t = BumpTerm()
    agents = list(s.cars)
    t.reset(agents, s, {})
    s.cars["blue-0"].bump_victim_id = "orange-0"
    s.cars["blue-1"].bump_victim_id = "blue-0"                              # coéquipier : rien
    r = t.get(agents, s, {})
    assert r["blue-0"] == 1.0 and r["blue-1"] == 0.0
    assert t.get(agents, s, {})["blue-0"] == 0.0                            # même contact, délai de RocketSim
    s.cars["blue-0"].bump_victim_id = None
    t.get(agents, s, {})
    s.cars["blue-0"].bump_victim_id = "orange-0"
    assert t.get(agents, s, {})["blue-0"] == 1.0                            # nouveau contact


def test_stage_weights_override_and_union_of_terms():
    cfg = load_config(["s10"])
    rf = make_reward(cfg)
    names = {t.name for t in rf.terms}
    # termes des paliers suivants construits dès le départ (aerial_touch, kickoff_win), poids nuls retirés
    assert {"strong_touch", "demo", "bump", "aerial_touch", "kickoff_win"} <= names and "double_commit" not in names
    assert rf.current_weight("touch", 0) == 0.5 and rf.current_weight("aerial_touch", 0) == 0.0
    s = make_state(1)
    agents = list(s.cars)
    rf.reset(agents, s, {})
    s.cars["blue-0"].ball_touches = 1
    base = rf.get_rewards(agents, s, {}, {}, {"global_step": 0})["blue-0"]
    no_touch = dict(cfg["rewards"]["weights"], touch=0.0)
    rf.reset(agents, s, {})
    over = rf.get_rewards(agents, s, {}, {}, {"global_step": 0, "reward_weights": no_touch})["blue-0"]
    assert np.isclose(base - over, 0.5, atol=1e-5)


def test_flip_speed_threshold_from_config_pays_slow_flips():
    """s10 : un flip qui fait passer de 300 à 800 uu/s vers la balle est payé (seuil 300), pas au seuil de s7e."""
    from test_s7e import _flip_run
    from rlagent.env.rewards import FlipSpeedTerm
    speeds = [300] * 4 + [500, 700, 800, 800, 800, 800]
    assert sum(_flip_run(speeds, flip_at=4)) == 0.0
    assert sum(_flip_run(speeds, flip_at=4, min_speed=300.0)) == pytest.approx(1.0)
    rf = make_reward(load_config(["s10"]))
    term = next(t for t in rf.terms if isinstance(t, FlipSpeedTerm))
    assert term.min_speed == 300.0


# --------------------------------------------------------------------------- curriculum
def _curr(**kw):
    stages = [{"name": "a", "advance": {"touches": 10, "mech/x": {"max": 1.0}}, "min_steps": 100},
              {"name": "b", "rewards": {"touch": 0.0, "new": 2.0}, "advance": {"g": 1}, "ppo": {"lr_actor": 1e-4}},
              {"name": "c", "ppo": {"ent_coef": 0.005}}]
    return SkillCurriculum({"touch": 1.0, "goal": 10.0}, stages, ema=1.0, patience=3, **kw)


def test_curriculum_advances_after_patience_and_min_steps():
    c = _curr(blend_steps=100)
    ok = {"touches": 12, "mech/x": 0.5}
    assert c.update(ok, 50) is None and c.streak == 0                       # min_steps pas atteint
    assert c.update({"touches": 12, "mech/x": 2.0}, 120) is None and c.streak == 0   # borne max dépassée
    for step in (130, 140):
        assert c.update(ok, step) is None
    assert "palier 2/3" in c.update(ok, 150)
    assert c.index == 1 and c.ppo_overrides() == {"lr_actor": 1e-4}
    # transition linéaire des poids
    assert c.weights(150) == pytest.approx({"touch": 1.0, "goal": 10.0, "new": 0.0})
    assert c.weights(200) == pytest.approx({"touch": 0.5, "goal": 10.0, "new": 1.0})
    assert c.weights(1000) == {"touch": 0.0, "goal": 10.0, "new": 2.0}
    for step in range(1000, 1040, 10):
        c.update({"g": 2}, step)
    assert c.index == 2 and c.ppo_overrides() == {"lr_actor": 1e-4, "ent_coef": 0.005}
    assert c.update({"g": 2}, 2000) is None                                 # dernier palier : on y reste


def test_curriculum_state_roundtrip():
    c = _curr()
    for step in (200, 210, 220):
        c.update({"touches": 11, "mech/x": 0.1}, step)
    d = json.loads(json.dumps(c.state_dict()))
    c2 = _curr()
    c2.load_state_dict(d)
    assert c2.index == c.index == 1 and c2.weights(225) == c.weights(225)


# --------------------------------------------------------------------------- échelle Elo
def test_ladder_recovers_strength_order_and_anchor(tmp_path):
    rng = np.random.default_rng(0)
    true = [0.0, 300.0, 550.0, 700.0]              # 4 versions successives, de plus en plus fortes
    anchor = 600.0
    lad = Ladder(tmp_path / "ladder.json")
    lad.add_version("v0", 0)
    for k in range(1, len(true)):
        res = []
        for _ in range(400):                       # la version k (en apprentissage) contre les précédentes
            j = int(rng.integers(0, k))
            p = 1 / (1 + 10 ** ((true[j] - true[k]) / 400))
            res.append((f"v{j}", float(rng.random() < p)))
            p_a = 1 / (1 + 10 ** ((anchor - true[k]) / 400))
            res.append(("anchor:s6", float(rng.random() < p_a)))
        lad.record(res, anchors={"anchor:s6"})
        lad.add_version(f"v{k}", 1000 * k)
    r = lad.fit()
    assert r["v0"] == 0.0
    assert r["v1"] < r["v2"] < r["v3"]
    assert abs(r["v3"] - 700) < 120 and abs(r["anchor:s6"] - anchor) < 120
    m = lad.metrics({"anchor:s6": "s6"})
    assert m["ladder/elo"] == pytest.approx(r["v3"]) and m["ladder/gap/s6"] == pytest.approx(r["v3"] - r["anchor:s6"])
    lad.save()
    again = Ladder(tmp_path / "ladder.json")
    assert again.fit() == pytest.approx(r)


def test_ladder_current_then_frozen_and_unknown_ignored(tmp_path):
    lad = Ladder(tmp_path / "l.json", min_games=5)
    lad.add_version("v0", 0)
    lad.record([("v0", 1.0)] * 9 + [("v0", 0.5), ("actor_from_other_run", 1.0)])
    lad.fit()
    assert lad.n_games(CURRENT) == 10 and "actor_from_other_run" not in lad.players
    assert lad.metrics()["ladder/elo_live"] > 200
    lad.add_version("v1", 10)
    assert lad.n_games(CURRENT) == 0 and lad.n_games("v1") == 10
    assert lad.fit()["v1"] > 200


# --------------------------------------------------------------------------- réseaux
torch = pytest.importorskip("torch")


def test_mlp_models_all_formats_and_pool_roundtrip():
    from rlagent.env.actions import N_ACTIONS
    from rlagent.env.obs import obs_dim
    from rlagent.models.attention import actor_from_model_cfg, actor_self_dim, build_models
    from rlagent.models.mlp import MLPActor, MLPCritic
    cfg = load_config(["s10"])
    actor, critic = build_models(cfg, N_ACTIONS)
    assert isinstance(actor, MLPActor) and isinstance(critic, MLPCritic)
    x = torch.randn(7, obs_dim(True))
    a, lp = actor.act(x)
    assert a.shape == (7,) and lp.shape == (7,) and critic(x).shape == (7,)
    # politique de départ quasi uniforme
    with torch.no_grad():
        assert float(actor.dist(x).entropy().mean()) > 4.45
    clone = actor_from_model_cfg(cfg["model"], N_ACTIONS)
    clone.load_state_dict(actor.state_dict())
    assert torch.allclose(clone(x), actor(x)) and actor_self_dim(clone) == cfg["model"]["self_dim"]
    # observation plus riche (agent plus récent) : fit_obs retire les colonnes en trop
    rich = torch.randn(3, obs_dim(True, True, True))
    assert actor(rich).shape == (3, N_ACTIONS)


def test_stacked_mlp_matches_each_actor():
    from rlagent.env.actions import N_ACTIONS
    from rlagent.env.obs import obs_dim
    from rlagent.models.attention import actor_from_model_cfg
    from rlagent.models.stacked import StackedActors, stack_signature
    cfg = load_config(["s10"])
    actors = []
    for seed in range(3):
        torch.manual_seed(seed)
        a = actor_from_model_cfg(cfg["model"], N_ACTIONS).eval()
        torch.nn.init.normal_(a.net[-1].weight, std=0.5)
        actors.append(a)
    sig = stack_signature(actors[0])
    assert sig is not None and sig == stack_signature(actors[1])
    assert stack_signature(actors[0].train()) is None
    actors[0].eval()
    st = StackedActors(actors)
    x = torch.randn(3, 5, obs_dim(True))
    out = st.logits(x)
    for k, a in enumerate(actors):
        assert torch.allclose(out[k], a(x[k]), atol=1e-5)


# --------------------------------------------------------------------------- profil complet (moteur factice)
def test_s10_profile_trains_end_to_end_with_ladder_and_curriculum(tmp_path):
    from rlagent.train import find_latest_checkpoint, train
    cfg = load_config(["debug", "s10"])
    cfg["env"]["engine"] = "fake"
    cfg["env"]["no_touch_timeout_s"] = 5
    cfg["env"]["episode_timeout_s"] = 20
    cfg["model"].update(actor_hidden=[64, 64], critic_hidden=[64, 64])
    cfg["ppo"].update(batch_steps=2000, minibatch_size=1000, device_chunk=1000, batch_schedule=None)
    cfg["train"].update(n_workers=2, total_steps=8000, checkpoint_every_steps=4000)
    cfg["selfplay"].update(pool_every_steps=2000, hist_prob=0.5, anchors=[])
    # palier 1 franchi tout de suite : on vérifie le passage au palier 2 et l'envoi des poids aux workers
    cfg["curriculum"]["stages"][0].update(advance={"touches_per_min": -1.0}, min_steps=0)
    cfg["curriculum"]["patience"] = 1
    cfg["curriculum"]["blend_steps"] = 2000
    cfg["run_name"] = "pytest_s10"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train(cfg, run_dir)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["global_step"] >= 8000 and all(np.isfinite(l["learn/policy_loss"]) for l in lines)
    assert lines[0]["curriculum/stage"] == 1.0 and lines[-1]["curriculum/blend"] == 1.0
    assert "learn/si_rot_deg" in lines[-1]
    # version de départ (step 0) dans le pool + échelle Elo remplie
    pool = sorted(p.name for p in (run_dir / "pool").glob("actor_*.pt"))
    assert pool[0] == "actor_0000000000000.pt" and len(pool) >= 4
    lad = json.loads((run_dir / "ladder.json").read_text())
    assert lad["players"]["actor_0000000000000"] == 0 and lad["ratings"]["actor_0000000000000"] == 0.0
    assert any("ladder/elo" in l for l in lines)
    # reprise : palier et échelle conservés
    cfg["train"]["total_steps"] = 10000
    train(cfg, run_dir, find_latest_checkpoint(run_dir))
    lines2 = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines2[-1]["global_step"] >= 10000 and lines2[-1]["curriculum/stage"] == 1.0
    assert len(json.loads((run_dir / "ladder.json").read_text())["players"]) >= len(lad["players"])


def test_shell_scripts_have_lf_line_endings():
    """Les scripts bash tournent sur Kaggle / Colab (Linux) : un CRLF (copie Windows réécrite) les casse
    (« $'\r': command not found »). Le zip de code est fait depuis la copie de travail, pas depuis git."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for sh in list((root / "cloud").glob("*.sh")) + list((root / "scripts").glob("*.sh")):
        assert b"\r\n" not in sh.read_bytes(), sh.name


def test_precision_stage_rebalances_toward_goal():
    """Palier 4 (25/10) : moins de « vite à la balle, frapper fort », plus de tirs cadrés, kickoffs et vraies aériennes."""
    from rlagent.env.rewards import AerialTouchTerm
    cfg = load_config(["kaggle_tpu", "s10"])
    c = SkillCurriculum.from_cfg(cfg)
    assert c.names == ["bases", "jeu", "avance", "precision"]
    w3, w4 = c.stage_weights[2], c.stage_weights[3]
    assert w4["vel_player_to_ball"] < w3["vel_player_to_ball"] and w4["strong_touch"] < w3["strong_touch"]
    assert w4["shot"] > 0 and w4["vel_ball_to_goal"] > w3["vel_ball_to_goal"] and w4["kickoff_win"] > w3["kickoff_win"]
    assert c.stages[2]["advance"] and c.ppo_overrides() == {}
    rf = make_reward(cfg)
    assert {"shot", "air_approach"} <= {t.name for t in rf.terms}
    assert next(t for t in rf.terms if isinstance(t, AerialTouchTerm)).min_z == 500.0


def test_best_version_is_saved_and_follows_the_ladder(tmp_path):
    from rlagent.train import save_best_version
    (tmp_path / "pool").mkdir()
    lad = Ladder(tmp_path / "ladder.json")
    lad.add_version("actor_0", 0)
    for k, wins in ((1, 9), (2, 4)):          # v1 bat largement v0 ; v2 perd contre v1
        (tmp_path / "pool" / f"actor_{k}.pt").write_bytes(bytes([k]))
        opp = f"actor_{k - 1}"
        lad.record([(opp, 1.0)] * wins + [(opp, 0.0)] * (10 - wins))
        lad.add_version(f"actor_{k}", k)
    lad.fit()
    assert save_best_version(tmp_path, lad) == "actor_1"
    assert (tmp_path / "best_actor.pt").read_bytes() == bytes([1])
    m = lad.metrics()
    assert m["ladder/best_step"] == 1.0 and m["ladder/drop"] > 0


def test_kaggle_output_download_resumes_after_network_cut(tmp_path, monkeypatch):
    """25/10 : une coupure au milieu des ~2 Go du pool faisait tout perdre. Reprise : le fichier tronqué est retiré,
    les fichiers complets ne sont pas retéléchargés (force=False après le 1er essai)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud"))
    import kaggle_sync
    monkeypatch.setattr(kaggle_sync.time, "sleep", lambda s: None)
    calls = []

    class FakeApi:
        def kernels_output(self, ref, dest, file_pattern=None, force=False):
            calls.append(force)
            d = Path(dest)
            d.mkdir(parents=True, exist_ok=True)
            (d / "a.pt").write_bytes(b"complet")
            if len(calls) == 1:
                (d / "b.pt").write_bytes(b"tron")
                raise ConnectionError("IncompleteRead")
            assert not (d / "b.pt").exists()          # fichier tronqué retiré avant la reprise
            (d / "b.pt").write_bytes(b"complet")
            return [str(d / "a.pt"), str(d / "b.pt")], None

    files = kaggle_sync.download_output(FakeApi(), "u/k", tmp_path / "out")
    assert calls == [True, False] and len(files) == 2
    assert (tmp_path / "out" / "b.pt").read_bytes() == b"complet"
