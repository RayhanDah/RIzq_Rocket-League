"""Banc d'essai hors ligne de l'optimiseur (phase 0 du bilan du 22/10/2026).

Collecte une fois N steps avec un checkpoint, exactement comme l'entraînement (workers, pool, exercices), puis rejoue
UNE itération d'update (`ppo_update`, le code d'entraînement tel quel) sur des copies du modèle, pour une grille de
variantes : lr de l'acteur × état d'Adam × taille de lot. Rien n'est écrit dans runs/ (pool copié dans le dossier de
sortie) et le checkpoint n'est pas modifié.

États d'Adam :
- `herite` : état du checkpoint (ce que font les démarrages à chaud et les reprises) ;
- `neuf`   : Adam neuf, moments estimés sur quelques minibatchs à lr = 0 (pas de saut du 1er pas) ;
- `normes` : comme `neuf`, après avoir remis la norme des couches suivies d'une LayerNorm à leur valeur
  d'initialisation (acteur et critique). Ces couches sont invariantes d'échelle : la politique ne change pas
  (écart vérifié et journalisé), seul le pas effectif de l'optimiseur est rétabli.
- `normes_herite` : normes remises et état d'Adam du checkpoint, moments remis à la même échelle (m / c, v / c²).

Mesures, sur des trajectoires de validation jamais vues par l'update :
- `kl` : KL(avant ‖ après) exact sur les 90 actions ; `kl_k3` : estimateur de learn/kl_final sur l'action jouée ;
- `gain` : objectif de substitution E[(r − 1)·Â] (amélioration prévue par échantillonnage d'importance ; > 0 :
  l'update généralise, ≈ 0 : il a surtout suivi du bruit), avec erreur type par trajectoire (`gain_se`) ;
- `d_entropy`, `rot_deg` (rotation des couches normalisées), `rel_disp` (‖Δθ‖ / ‖θ‖ de l'acteur).

    python scripts/bench_optimizer.py --run runs/s6_spo_kaggle --steps 2000000 --out <dossier> --device xpu
    (--reuse : réutilise la collecte déjà présente dans <dossier>)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlagent.algo.gae import RunningStats
from rlagent.algo.learner import build_batch, make_optimizer, ppo_update
from rlagent.config import _normalize
from rlagent.env.actions import N_ACTIONS
from rlagent.models.attention import build_models

# Linear suivies d'une LayerNorm (models/attention.py, mlp) : invariantes à l'échelle de (W, b)
SI_LAYERS = ("encoder.self_embed.0", "encoder.self_embed.3", "encoder.other_embed.0", "encoder.other_embed.3",
             "head.0.0", "head.0.3")
FIELDS = ("obs", "actions", "log_probs", "advantages", "returns", "values", "seg")


def si_vector(model: torch.nn.Module) -> torch.Tensor:
    parts = []
    for name in SI_LAYERS:
        lin = model.get_submodule(name)
        parts += [lin.weight.detach().float().cpu().flatten(), lin.bias.detach().float().cpu().flatten()]
    return torch.cat(parts)


@torch.no_grad()
def reset_norms(model: torch.nn.Module, ref: torch.nn.Module) -> dict:
    """Norme de chaque couche invariante d'échelle ramenée à celle d'un réseau neuf de même architecture.
    Renvoie {paramètre: facteur appliqué}."""
    factors = {}
    for name in SI_LAYERS:
        lin, lin0 = model.get_submodule(name), ref.get_submodule(name)
        n = math.sqrt(float(lin.weight.pow(2).sum() + lin.bias.pow(2).sum()))
        n0 = math.sqrt(float(lin0.weight.pow(2).sum() + lin0.bias.pow(2).sum()))
        lin.weight.mul_(n0 / n)
        lin.bias.mul_(n0 / n)
        factors[lin.weight] = factors[lin.bias] = n0 / n
    return factors


@torch.no_grad()
def rescale_adam_state(opt: torch.optim.Optimizer, factors: dict) -> None:
    """Après w → c·w sur une couche invariante d'échelle, son gradient devient g / c : on met les moments d'Adam à la
    même échelle (m / c, v / c²). Le pas d'Adam garde sa taille, mais il est c fois plus grand relativement aux poids."""
    for p, c in factors.items():
        st = opt.state.get(p)
        if st:
            st["exp_avg"].div_(c)
            st["exp_avg_sq"].div_(c * c)


def collect(cfg, ck, run_dir: Path, out: Path, n_total: int, workers: int, envs: int) -> dict:
    from rlagent.algo.collector import make_collector
    from rlagent.selfplay.pool import OpponentPool
    from rlagent.train import DrillCurriculum, build_actor_from_model_cfg

    cfg = copy.deepcopy(cfg)
    cfg["train"].update(async_collect=False, collect_mode="central", collect_device="cpu",
                        n_workers=workers, envs_per_worker=envs)
    actor, critic = build_models(cfg, N_ACTIONS)
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    ret_stats = RunningStats()
    ret_stats.load_state_dict(ck["ret_stats"])
    step = int(ck["global_step"])
    cur = DrillCurriculum.from_cfg(cfg)
    if cur is not None and ck.get("curriculum"):
        cur.load_state_dict(ck["curriculum"])
    pool_dir = out / "pool_copy"                       # copie : rien n'est écrit dans runs/
    shutil.rmtree(pool_dir, ignore_errors=True)
    shutil.copytree(run_dir / "pool", pool_dir)
    sp = cfg["selfplay"]
    pool = OpponentPool(pool_dir, recent=int(sp["pool_recent"]), seed=1234, pfsp=bool(sp.get("pfsp", False)),
                        pfsp_uniform=float(sp.get("pfsp_uniform", 0.3)), pfsp_window=float(sp.get("pfsp_window", 50)),
                        anchors=sp.get("anchors") or [], anchor_prob=float(sp.get("anchor_prob", 0.2)),
                        anchor_root=ROOT)
    collector = make_collector(cfg, workers, 1234, pool=pool, build_actor_fn=build_actor_from_model_cfg)
    cpu = torch.device("cpu")
    parts = {k: [] for k in FIELDS}
    got, seg0, t0 = 0, 0, time.time()
    try:
        collector.set_global_step(step)
        if cur is not None:
            collector.set_info({"drill_levels": dict(cur.levels)})
        while got < n_total:
            trajs, _, n, _ = collector.collect(actor, min(200_000, n_total - got), cpu)
            batch = build_batch(trajs, critic, ret_stats, cfg, cpu)
            for k in FIELDS[:-1]:
                parts[k].append(batch[k])
            parts["seg"].append(np.repeat(np.arange(len(trajs)) + seg0, [len(t) for t in trajs]).astype(np.int32))
            seg0 += len(trajs)
            got += n
            print(f"[collecte] {got:,} steps, {seg0:,} morceaux de trajectoire, {time.time() - t0:.0f} s", flush=True)
    finally:
        collector.close()
    data = {k: np.concatenate(v) for k, v in parts.items()}
    for k, v in data.items():
        np.save(out / f"data_{k}.npy", v)
    meta = {"global_step": step, "n_steps": len(data["actions"]), "n_segments": seg0,
            "collect_s": round(time.time() - t0, 1)}
    (out / "data_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return data


def draw_segments(rng, seg_ids, seg_start, seg_len, n_steps, exclude=None):
    """Morceaux de trajectoire entiers (comme un vrai lot) jusqu'à ≥ n_steps ; renvoie (indices, morceaux pris)."""
    pool = seg_ids if exclude is None else np.setdiff1d(seg_ids, exclude)
    order = rng.permutation(pool)
    k = int(np.searchsorted(np.cumsum(seg_len[order]), n_steps)) + 1
    take = order[:k]
    return np.concatenate([np.arange(seg_start[j], seg_start[j] + seg_len[j]) for j in take]), take


@torch.no_grad()
def log_softmax_all(actor, obs: np.ndarray, dev, chunk: int) -> torch.Tensor:
    actor.eval()
    out = [torch.log_softmax(actor(torch.as_tensor(obs[s:s + chunk]).to(dev)).float(), -1).cpu()
           for s in range(0, len(obs), chunk)]
    actor.train()
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None, help="défaut : dernier checkpoint du run")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--val-steps", type=int, default=300_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--configs", default="50000:25000:4,300000:100000:4", help="lot:minibatch:epochs, …")
    ap.add_argument("--states", default="herite,neuf,normes")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--warmup-mb", type=int, default=8, help="minibatchs à lr = 0 pour les Adam neufs")
    ap.add_argument("--chunk", type=int, default=25_000)
    ap.add_argument("--results", default="results.json", help="nom du fichier de résultats dans --out")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    _normalize(cfg)
    ckpt = args.ckpt or max((args.run / "checkpoints").glob("ckpt_*.pt"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(f"[banc] checkpoint {ckpt} ({int(ck['global_step']):,} steps)", flush=True)

    torch.set_num_threads(2)                           # inférence de collecte (cf. laptop_xpu)
    if args.reuse and (out / "data_meta.json").exists():
        data = {k: np.load(out / f"data_{k}.npy") for k in FIELDS}
        print(f"[banc] collecte réutilisée : {len(data['actions']):,} steps", flush=True)
    else:
        data = collect(cfg, ck, args.run, out, args.steps, args.workers, args.envs)
    torch.set_num_threads(8)

    dev = torch.device(args.device)
    seg = data["seg"]
    seg_len = np.bincount(seg)
    seg_start = np.concatenate([[0], np.cumsum(seg_len)[:-1]])
    seg_ids = np.arange(len(seg_len))
    rng = np.random.default_rng(0)
    val_idx, val_segs = draw_segments(rng, seg_ids, seg_start, seg_len, args.val_steps)
    train_segs = np.setdiff1d(seg_ids, val_segs)
    obs_v, act_v = data["obs"][val_idx], torch.as_tensor(data["actions"][val_idx])
    adv_v = data["advantages"][val_idx]
    adv_v = torch.as_tensor((adv_v - adv_v.mean()) / (adv_v.std() + 1e-8), dtype=torch.float32)
    seg_v = torch.as_tensor(np.searchsorted(val_segs[np.argsort(val_segs)], seg[val_idx]))
    print(f"[banc] validation : {len(val_idx):,} steps ({len(val_segs)} morceaux) ; update : "
          f"{len(data['actions']) - len(val_idx):,} steps", flush=True)

    base_actor, base_critic = build_models(cfg, N_ACTIONS)
    base_actor.load_state_dict(ck["actor"])
    base_critic.load_state_dict(ck["critic"])
    torch.manual_seed(0)
    ref_actor, ref_critic = build_models(cfg, N_ACTIONS)  # réseau neuf : normes de référence
    lp0 = log_softmax_all(base_actor.to(dev), obs_v, dev, args.chunk)
    base_actor.cpu()
    p0 = lp0.exp()
    ent0 = float(-(p0 * lp0).sum(-1).mean())
    a_lp0 = lp0.gather(1, act_v[:, None]).squeeze(1)
    si0 = si_vector(base_actor)
    probe = obs_v[:8192]

    def make(state: str):
        actor, critic = copy.deepcopy(base_actor), copy.deepcopy(base_critic)
        info, factors = {}, {}
        if state.startswith("normes"):
            before = log_softmax_all(actor.to(dev), probe, dev, args.chunk)
            factors = {**reset_norms(actor.cpu(), ref_actor), **reset_norms(critic, ref_critic)}
            after = log_softmax_all(actor.to(dev), probe, dev, args.chunk)
            info["norm_reset_max_dlogp"] = float((after - before).abs().max())
            info["norm_reset_kl"] = float((before.exp() * (before - after)).sum(-1).mean())
            info["si_norm_before"], info["si_norm_after"] = float(si0.norm()), float(si_vector(actor).norm())
        actor.to(dev)
        critic.to(dev)
        opt = make_optimizer(actor, critic, cfg)
        if state in ("herite", "normes_herite"):
            opt.load_state_dict(copy.deepcopy(ck["optimizer"]))   # sur CPU, les tenseurs seraient partagés
        if state == "normes_herite":
            rescale_adam_state(opt, factors)
        return actor, critic, opt, info

    def batch_of(idx):
        return {k: data[k][idx] for k in ("obs", "actions", "log_probs", "advantages", "returns", "values")}

    def ucfg(B, mb, ep):
        c = copy.deepcopy(cfg)
        c["ppo"].update(batch_steps=B, minibatch_size=mb, epochs=ep, lr_schedule="fixed", device_chunk=args.chunk)
        return c

    results = []
    lrs = [float(x) for x in args.lrs.split(",")]
    confs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    t_all = time.time()
    for B, mb, ep in confs:
        for seed in (int(s) for s in args.seeds.split(",")):
            srng = np.random.default_rng(100 + seed)
            upd_idx, upd_segs = draw_segments(srng, train_segs, seg_start, seg_len, B)
            wu_idx, _ = draw_segments(srng, train_segs, seg_start, seg_len, args.warmup_mb * mb, exclude=upd_segs)
            for state in args.states.split(","):
                for lr in lrs:
                    t0 = time.time()
                    torch.manual_seed(seed)
                    actor, critic, opt, info = make(state)
                    si_start = si_vector(actor)        # après remise des normes le cas échéant
                    th_start = torch.cat([p.detach().cpu().flatten() for p in actor.parameters()])
                    lr_c = float(cfg["ppo"]["lr_critic"])
                    if state not in ("herite", "normes_herite"):   # moments d'Adam estimés sans bouger les poids
                        for g in opt.param_groups:
                            g["lr"] = 0.0
                        ppo_update(actor, critic, opt, batch_of(wu_idx), ucfg(len(wu_idx), mb, 1), dev)
                    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr, lr_c
                    stats = ppo_update(actor, critic, opt, batch_of(upd_idx), ucfg(B, mb, ep), dev)
                    lp1 = log_softmax_all(actor, obs_v, dev, args.chunk)
                    kl = float((p0 * (lp0 - lp1)).sum(-1).mean())
                    a_lp1 = lp1.gather(1, act_v[:, None]).squeeze(1)
                    lr_ = a_lp1 - a_lp0
                    r = lr_.exp()
                    x = (r - 1.0) * adv_v
                    gain = float(x.mean())
                    sums = torch.zeros(int(seg_v.max()) + 1).index_add_(0, seg_v, x)
                    cnt = torch.zeros_like(sums).index_add_(0, seg_v, torch.ones_like(x))
                    gain_se = float(((sums - cnt * gain).pow(2).sum()).sqrt() / len(x))
                    ent1 = float(-(lp1.exp() * lp1).sum(-1).mean())
                    a_cpu = actor.cpu()
                    si1 = si_vector(a_cpu)
                    cosang = float(torch.dot(si_start, si1) / (si_start.norm() * si1.norm()))
                    th1 = torch.cat([p.detach().flatten() for p in a_cpu.parameters()])
                    row = {"batch": B, "minibatch": mb, "epochs": ep, "steps": stats["n_minibatches"], "seed": seed,
                           "state": state, "lr": lr, "kl": kl, "kl_k3": float(((r - 1) - lr_).mean()),
                           "kl_final_batch": stats["kl_final"], "gain": gain, "gain_se": gain_se,
                           "d_entropy": ent1 - ent0, "rot_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosang)))),
                           "rel_disp": float((th1 - th_start).norm() / th_start.norm()),
                           "clip_frac": stats["clip_frac"], "warmup_samples": 0 if state in ("herite", "normes_herite") else len(wu_idx),
                           "secs": round(time.time() - t0, 1), **info}
                    results.append(row)
                    print(f"[update] lot {B:>7,} mb {mb:>7,} ×{ep} graine {seed} {state:<6} lr {lr:.0e} : "
                          f"KL {kl:.5f}  gain {gain:+.5f} ± {gain_se:.5f}  ΔH {ent1 - ent0:+.4f}  "
                          f"rot {row['rot_deg']:.3f}°  ({row['secs']} s)", flush=True)
                    (out / args.results).write_text(json.dumps({"checkpoint": str(ckpt), "entropy0": ent0,
                                                                  "rows": results}, indent=1), encoding="utf-8")
    print(f"[banc] terminé en {time.time() - t_all:.0f} s → {out / args.results}", flush=True)


if __name__ == "__main__":
    main()
