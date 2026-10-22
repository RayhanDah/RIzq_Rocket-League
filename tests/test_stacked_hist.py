"""Inférence groupée des versions du pool (`models/stacked.py`, `Rollout.act`) : mêmes logits / mêmes actions que la
boucle d'origine (une inférence par version)."""
import numpy as np
import pytest

from rlagent.config import load_config

torch = pytest.importorskip("torch")

from rlagent.algo.collector import EnvSlot, Rollout  # noqa: E402
from rlagent.algo.worker import EpisodeStart  # noqa: E402
from rlagent.env.actions import N_ACTIONS  # noqa: E402
from rlagent.env.obs import MAX_OTHERS, OTHER_DIM  # noqa: E402
from rlagent.models.attention import SELF_DIMS, actor_from_model_cfg, fit_obs  # noqa: E402
from rlagent.models.stacked import StackedActors, pad_stack, stack_signature  # noqa: E402

OBS_SELF = SELF_DIMS[-1]                                 # env le plus riche ; les versions plus courtes passent par fit_obs
OBS_DIM = OBS_SELF + MAX_OTHERS * OTHER_DIM + MAX_OTHERS


def _version(seed, self_dim=OBS_SELF):
    """Acteur aléatoire à logits bien séparés (pas d'ex aequo à 1e-6 près pour l'argmax)."""
    torch.manual_seed(seed)
    cfg = dict(load_config(["debug"])["model"], self_dim=self_dim)
    actor = actor_from_model_cfg(cfg, N_ACTIONS)
    torch.nn.init.normal_(actor.head[-1].weight, std=1.0)
    return actor.eval()


def _obs(n, rng):
    o = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    o[:, -MAX_OTHERS:] = np.arange(MAX_OTHERS)[None] < rng.integers(0, MAX_OTHERS + 1, (n, 1))
    return o


class RowwiseArgmax(torch.nn.Module):
    """Version hors `Actor` (architecture inconnue) : doit passer par la boucle d'origine."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    @torch.no_grad()
    def act(self, obs):
        logits = self.actor(obs)
        a = logits.argmax(-1)
        return a, torch.distributions.Categorical(logits=logits).log_prob(a)


def test_stacked_logits_match_actor():
    rng = np.random.default_rng(0)
    for self_dim in SELF_DIMS:
        actors = [_version(s, self_dim) for s in range(4)]
        sizes = [1, 5, 2, 300]                           # 300 ≥ SKIP_PADDING_MIN_BATCH : chemin skip_padding d'Actor
        obs = [list(_obs(n, rng)) for n in sizes]
        x = fit_obs(torch.as_tensor(pad_stack(obs)), self_dim)
        out = StackedActors(actors).logits(x)
        assert out.shape == (4, max(sizes), N_ACTIONS)
        with torch.no_grad():
            for k, (a, o) in enumerate(zip(actors, obs)):
                ref = a(torch.as_tensor(np.stack(o)))
                # float32 : l'ordre des sommes diffère (bmm groupé) ; 141 entrées N(0, 1) → écarts jusqu'à ~1e-5
                torch.testing.assert_close(out[k, :len(o)], ref, rtol=1e-5, atol=3e-5)


def test_stack_signature():
    a, b, c = _version(0), _version(1), _version(2, SELF_DIMS[0])
    assert stack_signature(a) is not None and stack_signature(a) == stack_signature(b)
    assert stack_signature(c) != stack_signature(a)          # autre bloc « soi » : autre pile
    assert stack_signature(RowwiseArgmax(a)) is None
    assert stack_signature(a.train()) is None


def _rollout(versions, stack_hist, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(versions)
    starts = []
    for i in range(24):
        n = int(rng.integers(1, 4))
        starts.append(EpisodeStart([f"a{j}" for j in range(2 * n)], np.repeat([0, 1], n), _obs(2 * n, rng)))
    picks = iter([i % (len(ids) + 1) for i in range(len(starts))])       # chaque version jouée ; len(ids) = self-play

    def new_slot(start):
        slot = EnvSlot(episode=start)
        p = next(picks)
        if p < len(ids):
            slot.hist_team, slot.hist_entry = p % 2, {"id": ids[p], "elo": 0.0}
        return slot

    return Rollout(starts, new_slot, stack_hist=stack_hist)


def test_rollout_stacked_matches_loop(monkeypatch):
    """Politique déterministe (tirage remplacé par l'argmax) : actions et log-probs identiques avec et sans pile."""
    monkeypatch.setattr(torch.distributions.Categorical, "sample",
                        lambda self, sample_shape=torch.Size(): self.logits.argmax(-1))
    versions = {f"v{s}": _version(s) for s in range(5)}
    versions.update({f"old{s}": _version(10 + s, SELF_DIMS[0]) for s in range(2)})   # ancres via fit_obs
    versions["solo"] = _version(20, SELF_DIMS[1])                                  # seule de son architecture
    versions["wrapped"] = RowwiseArgmax(_version(30))
    current = _version(99)
    out = {}
    for stack in (False, True):
        ro = _rollout(versions, stack)
        acts = [ro.act(range(24), current, torch.device("cpu"), lambda e: versions[e["id"]]) for _ in range(2)]
        logp = [(t.slot, t.agent, t.log_probs) for s in ro.slots for t in s.trajs.values()]
        out[stack] = (acts, logp, ro.stack_builds)
    (acts_loop, logp_loop, b0), (acts_stack, logp_stack, b1) = out[False], out[True]
    assert b0 == 0 and b1 >= 1                       # la pile a servi… et n'a pas été reconstruite au 2e appel
    assert b1 <= 2
    for a, b in zip(acts_loop, acts_stack):
        assert a.keys() == b.keys()
        for w in a:
            np.testing.assert_array_equal(a[w], b[w])
    assert logp_loop == logp_stack
