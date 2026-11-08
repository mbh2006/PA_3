"""N=32 signal analyses: length/score/correctness associations, quartiles, fusion."""

from __future__ import annotations

import numpy as np

from task3.scores import visconf_confidence, visconf_from_confidence

QUARTILE = 8  # 32 candidates -> 4 groups of 8


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    from scipy.stats import spearmanr

    rho, _ = spearmanr(x, y)
    return None if not np.isfinite(rho) else float(rho)


def quartile_stats(score: np.ndarray, correct: np.ndarray) -> float | None:
    if len(score) < 2 * QUARTILE or np.all(score == score[0]):
        return None
    order = np.argsort(-score, kind="mergesort")
    top = correct[order[:QUARTILE]].mean()
    bottom = correct[order[-QUARTILE:]].mean()
    return float(top - bottom)


def within_question(bundle, ids: list[str], *, n_tokens_only: bool = False) -> dict:
    """Full-pool (N=32) within-question associations, averaged across questions."""
    by_id = bundle.by_id()
    keys = ["length", "logp", "self_certainty"] + (["vision", "visconf"] if bundle.is_vlm else [])

    corrs: dict[str, list[float]] = {k: [] for k in keys}
    quart: dict[str, list[float]] = {k: [] for k in keys}
    shortest_correct: list[float] = []
    shortest_is_best_score: list[float] = []

    for sid in ids:
        q = by_id[sid]
        n = len(q.rollout_ids)
        idx = list(range(n))
        correct = q.correct.astype(float)
        scores = {
            "length": q.n_tokens.astype(float),
            "logp": q.logp_mean,
            "self_certainty": q.kl_mean,
        }
        if bundle.is_vlm:
            conf = visconf_confidence(q.kl_tokens, idx)
            scores["vision"] = q.vis
            scores["visconf"] = visconf_from_confidence(conf, q.vis)

        for k, s in scores.items():
            r = _spearman(np.asarray(s, dtype=float), correct)
            if r is not None:
                corrs[k].append(r)
            d = quartile_stats(np.asarray(s, dtype=float), correct)
            if d is not None:
                quart[k].append(d)

        short = int(np.argmin(q.n_tokens))
        shortest_correct.append(float(correct[short]))
        best = int(np.argmax(scores["self_certainty"]))
        shortest_is_best_score.append(float(short == best))

        if n_tokens_only:
            continue

    def pack(values: list[float]) -> dict:
        if not values:
            return {"n_defined": 0, "mean": None, "median": None, "sd": None}
        a = np.asarray(values, dtype=float)
        return {
            "n_defined": int(len(a)),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        }

    return {
        "spearman_with_correctness": {k: pack(v) for k, v in corrs.items()},
        "quartile_top_minus_bottom": {k: pack(v) for k, v in quart.items()},
        "shortest_rollout_accuracy": pack(shortest_correct),
        "shortest_equals_best_self_certainty": pack(shortest_is_best_score),
    }


def confidence_vision_correlations(bundle, ids: list[str]) -> dict:
    """Within-rollout corr(c_t, v_t) with full-pool VisConf calibration."""
    by_id = bundle.by_id()
    per_question, per_rollout_all = [], []
    for sid in ids:
        q = by_id[sid]
        idx = list(range(len(q.rollout_ids)))
        conf = visconf_confidence(q.kl_tokens, idx)
        rs = []
        for i, c in enumerate(conf):
            k = np.asarray(q.kl_tokens[i], dtype=np.float64)
            v = np.asarray(q.v_tokens[i], dtype=np.float64)
            # recompute c_t for this rollout from the pooled calibration
            pooled = np.concatenate([q.kl_tokens[j] for j in idx]).astype(np.float64)
            q05, q95 = np.quantile(pooled, [0.05, 0.95], method="linear")
            denom = max(float(q95 - q05), 1e-8)
            ct = np.clip((k - q05) / denom, 0.0, 1.0)
            if len(ct) < 3 or np.all(ct == ct[0]) or np.all(v == v[0]):
                continue
            r = float(np.corrcoef(ct, v)[0, 1])
            if np.isfinite(r):
                rs.append(r)
        if rs:
            per_question.append(float(np.mean(rs)))
            per_rollout_all.extend(rs)
    a = np.asarray(per_question, dtype=float) if per_question else np.array([])
    b = np.asarray(per_rollout_all, dtype=float) if per_rollout_all else np.array([])
    return {
        "n_questions_defined": int(len(a)),
        "n_rollouts_defined": int(len(b)),
        "question_mean_of_rollout_corr": {
            "mean": float(a.mean()) if len(a) else None,
            "median": float(np.median(a)) if len(a) else None,
            "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        },
        "rollout_pooled_corr": {
            "mean": float(b.mean()) if len(b) else None,
            "median": float(np.median(b)) if len(b) else None,
            "p10": float(np.quantile(b, 0.10)) if len(b) else None,
            "p90": float(np.quantile(b, 0.90)) if len(b) else None,
        },
    }


def fusion_identity(bundle, ids: list[str], *, max_questions: int | None = None) -> dict:
    """Check Z_token - Z == population Cov_t(c_t, v_t) for every rollout."""
    by_id = bundle.by_id()
    worst = 0.0
    n_checked = 0
    examples = []
    selected = ids if max_questions is None else ids[:max_questions]
    for sid in selected:
        q = by_id[sid]
        idx = list(range(len(q.rollout_ids)))
        pooled = np.concatenate([q.kl_tokens[j] for j in idx]).astype(np.float64)
        q05, q95 = np.quantile(pooled, [0.05, 0.95], method="linear")
        denom = max(float(q95 - q05), 1e-8)
        for i in range(len(idx)):
            k = np.asarray(q.kl_tokens[i], dtype=np.float64)
            v = np.asarray(q.v_tokens[i], dtype=np.float64)
            c = np.clip((k - q05) / denom, 0.0, 1.0)
            z_tok = float(np.mean(c * v))
            z = float(np.mean(c) * np.mean(v))
            cov = float(np.mean((c - c.mean()) * (v - v.mean())))
            worst = max(worst, abs(z_tok - z - cov))
            n_checked += 1
            if len(examples) < 3:
                examples.append({
                    "sample_id": sid, "rollout": i,
                    "z_token": z_tok, "z": z, "cov": cov, "abs_err": abs(z_tok - z - cov),
                })
    return {
        "n_rollouts_checked": n_checked,
        "max_abs_error": worst,
        "identity": "Z_token - Z = Cov_t(c_t, v_t) (population covariance)",
        "examples": examples,
    }


def save_trajectories(bundle, ids: list[str], out_path, *, max_per_type: int = 6) -> dict:
    """Save a fixed set of per-token (c_t, v_t) trajectories for inspection.

    Selection rule: the first question (in split order) with the highest positive and
    the highest negative rollout-level corr(c_t, v_t), plus the first question overall.
    Deterministic given the split and data.
    """
    import json

    by_id = bundle.by_id()
    scored = []
    for sid in ids:
        q = by_id[sid]
        idx = list(range(len(q.rollout_ids)))
        pooled = np.concatenate([q.kl_tokens[j] for j in idx]).astype(np.float64)
        q05, q95 = np.quantile(pooled, [0.05, 0.95], method="linear")
        denom = max(float(q95 - q05), 1e-8)
        best = (None, -2.0)
        for i in range(len(idx)):
            k = np.asarray(q.kl_tokens[i], dtype=np.float64)
            v = np.asarray(q.v_tokens[i], dtype=np.float64)
            c = np.clip((k - q05) / denom, 0.0, 1.0)
            if len(c) < 3 or np.all(c == c[0]) or np.all(v == v[0]):
                continue
            r = float(np.corrcoef(c, v)[0, 1])
            if np.isfinite(r) and r > best[1]:
                best = (i, r)
        if best[0] is not None:
            scored.append((sid, best[0], best[1]))

    chosen: list[tuple[str, int, str]] = []
    if scored:
        top = max(scored, key=lambda t: t[2])
        bottom = min(scored, key=lambda t: t[2])
        chosen.append((top[0], top[1], "highest_corr"))
        chosen.append((bottom[0], bottom[1], "lowest_corr"))
        chosen.append((ids[0], 0, "first_question"))
    seen, payload = set(), []
    for sid, rid, tag in chosen:
        if (sid, rid) in seen:
            continue
        seen.add((sid, rid))
        q = by_id[sid]
        idx = list(range(len(q.rollout_ids)))
        pooled = np.concatenate([q.kl_tokens[j] for j in idx]).astype(np.float64)
        q05, q95 = np.quantile(pooled, [0.05, 0.95], method="linear")
        denom = max(float(q95 - q05), 1e-8)
        k = np.asarray(q.kl_tokens[rid], dtype=np.float64)
        v = np.asarray(q.v_tokens[rid], dtype=np.float64)
        c = np.clip((k - q05) / denom, 0.0, 1.0)
        payload.append({
            "tag": tag,
            "sample_id": sid,
            "rollout_id": int(rid),
            "n_tokens": int(len(k)),
            "c_t": c.tolist(),
            "v_t": v.tolist(),
            "z_token": float(np.mean(c * v)),
            "z": float(np.mean(c) * np.mean(v)),
        })
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return {"n_trajectories": len(payload), "path": str(out_path)}
