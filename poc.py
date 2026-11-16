"""Graph-contract proof of concept on GPT-2 small (see PREREG.md for every choice).

Builds the direct-path gradient effective-connectivity graph (Definition 1 of the paper) over the
144 attention heads on IOI prompts, then tests H1 (rich-club ordering) and H2 (modularity against
the Wang et al. 2023 head classes) against nulls that keep degree, layer-pair edge counts and layer
order. Also validates 20 edges against finite path patching.

Usage:
  python poc.py --smoke      # tiny run, prints shapes and timings only, no test statistics
  python poc.py              # full pre-registered run, writes results/
"""
import argparse, json, math, time, hashlib
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import adjusted_mutual_info_score
from transformer_lens import HookedTransformer

L, H, DH = 12, 12, 64
N = L * H

TEMPLATES = [
    "Then, {N1} and {N2} went to the {PLACE}. {S} gave a {OBJECT} to",
    "When {N1} and {N2} went to the {PLACE}, {S} gave the {OBJECT} to",
    "After {N1} and {N2} went to the {PLACE}, {S} gave a {OBJECT} to",
    "Then, {N1} and {N2} had a long argument, and afterwards {S} said to",
    "While {N1} and {N2} were working at the {PLACE}, {S} gave a {OBJECT} to",
    "Friends {N1} and {N2} found a {OBJECT} at the {PLACE}. {S} gave it to",
    "Then, {N1} and {N2} were thinking about going to the {PLACE}. {S} wanted to give a {OBJECT} to",
    "When {N1} and {N2} got a {OBJECT} at the {PLACE}, {S} decided to give it to",
    "The {PLACE} {N1} and {N2} went to had a {OBJECT}. {S} gave it to",
    "Then, {N1} and {N2} had a lot of fun at the {PLACE}. {S} gave a {OBJECT} to",
    "Afterwards, {N1} and {N2} went to the {PLACE}. {S} gave a {OBJECT} to",
    "When {N1} and {N2} arrived at the {PLACE}, {S} gave a {OBJECT} to",
    "Then, {N1} and {N2} went to the {PLACE}, and {S} handed a {OBJECT} to",
    "While {N1} and {N2} were commuting to the {PLACE}, {S} gave a {OBJECT} to",
    "After the lunch, {N1} and {N2} went to the {PLACE}. {S} gave a {OBJECT} to",
]
NAMES = ["John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Lisa", "Michael", "Robert",
         "Daniel", "Jessica", "Sarah", "Kevin", "Paul", "Mark", "Jacob", "David", "Richard", "Laura",
         "Anna", "Emma", "Alice", "George", "Peter", "Kate", "Jane", "Steven", "Brian", "Eric", "Ryan",
         "Adam", "Chris", "Joe", "Mike", "Frank", "Rose", "Grace", "Ben"]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station", "park", "beach"]
OBJECTS = ["drink", "ring", "kiss", "bone", "basketball", "computer", "necklace", "snack", "apple", "book"]

LABELS = {  # Wang et al. (2023), 7 classes, 26 heads (PREREG.md)
    "name mover": ["9.6", "9.9", "10.0"],
    "negative name mover": ["10.7", "11.10"],
    "backup name mover": ["9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"],
    "S-inhibition": ["7.3", "7.9", "8.6", "8.10"],
    "induction": ["5.5", "5.8", "5.9", "6.9"],
    "duplicate token": ["0.1", "0.10", "3.0"],
    "previous token": ["2.2", "4.11"],
}

SEEDS = dict(edge_prompts=0, heldout_prompts=1, probes=2, nulls=3, bootstrap=4, label_perm=5,
             validation_edges=6, louvain_first=100)


def idx(lh):
    l, h = map(int, lh.split("."))
    return l * H + h


def name(i):
    return f"{i // H}.{i % H}"


# ---------------------------------------------------------------- prompts

def single_token(model, words):
    return [w for w in words if len(model.to_tokens(" " + w, prepend_bos=False)[0]) == 1]


def make_prompts(model, n, seed, exclude=frozenset()):
    rng = np.random.default_rng(seed)
    names = single_token(model, NAMES)
    places, objects = single_token(model, PLACES), single_token(model, OBJECTS)
    out, seen = [], set(exclude)
    while len(out) < n:
        t = int(rng.integers(len(TEMPLATES)))
        io, s, c1, c2, c3 = rng.choice(names, 5, replace=False)
        place, obj = rng.choice(places), rng.choice(objects)
        order = len(out) % 2  # 0: ABBA (IO first), 1: BABA (S first)
        key = (t, io, s, place, obj, order)
        if key in seen:
            continue
        seen.add(key)
        n1, n2 = (io, s) if order == 0 else (s, io)
        clean = TEMPLATES[t].format(N1=n1, N2=n2, S=s, PLACE=place, OBJECT=obj)
        corrupt = TEMPLATES[t].format(N1=c1, N2=c2, S=c3, PLACE=place, OBJECT=obj)  # ABC
        out.append(dict(t=t, clean=clean, corrupt=corrupt, io=" " + io, s=" " + s, key=key))
    return out, seen


def by_template(prompts):
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(p["t"], []).append(i)
    return groups


# ---------------------------------------------------------------- model pieces

def z_of_layer(model, layer, resid):
    """z of all heads of block `layer` from its input residual stream (manual, differentiable)."""
    blk = model.blocks[layer]
    x = blk.ln1(resid)
    a = blk.attn
    q = torch.einsum("btd,hde->bthe", x, a.W_Q) + a.b_Q
    k = torch.einsum("btd,hde->bthe", x, a.W_K) + a.b_K
    v = torch.einsum("btd,hde->bthe", x, a.W_V) + a.b_V
    scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(DH)
    T = resid.shape[1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=resid.device), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    pattern = scores.softmax(-1)
    return torch.einsum("bhqk,bkhe->bqhe", pattern, v)  # (B, T, H, DH)


def cache_run(model, toks):
    names = lambda n: n.endswith("hook_z") or n.endswith("hook_resid_pre")
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks, names_filter=names)
    return logits, cache


def ec_per_prompt(model, toks, n_probes, gen, dev):
    """Hutchinson estimate of ||J_ba||_F^2 (direct path) for every a, b; shape (B, N, N), [a, b]."""
    _, cache = cache_run(model, toks)
    B = toks.shape[0]
    ec = torch.zeros(B, N, N)
    W_O = model.W_O.detach()  # (L, H, DH, d_model)
    for lb in range(1, L):
        resid = cache[f"blocks.{lb}.hook_resid_pre"].detach()
        r = resid.repeat_interleave(n_probes * H, 0).requires_grad_(True)  # (B*P*H, T, D)
        z = z_of_layer(model, lb, r)  # (B*P*H, T, H, DH)
        u = torch.randn(z.shape, generator=gen, device="cpu").to(dev)
        hb = torch.arange(H, device=dev).repeat(B * n_probes)  # receiver head per row
        sel = torch.zeros(z.shape[0], 1, H, 1, device=dev)
        sel[torch.arange(z.shape[0]), 0, hb, 0] = 1.0
        (g,) = torch.autograd.grad((z * u * sel).sum(), r)  # (B*P*H, T, D) = J_b^T u
        g = g.detach()
        for la in range(lb):
            proj = torch.einsum("ntd,hed->nhte", g, W_O[la])  # M_a^T J_b^T u for the 12 heads of la
            sq = proj.pow(2).sum((2, 3))  # (B*P*H, H_a)
            sq = sq.view(B, n_probes, H, H).mean(1)  # (B, H_b, H_a)
            ec[:, la * H:(la + 1) * H, lb * H:(lb + 1) * H] = sq.transpose(1, 2).cpu()
        del r, z, g
    return ec


def logit_diff(model, logits, prompts, rows):
    io = torch.tensor([model.to_single_token(prompts[i]["io"]) for i in rows])
    s = torch.tensor([model.to_single_token(prompts[i]["s"]) for i in rows])
    last = logits[:, -1, :].cpu()
    return (last[torch.arange(len(rows)), io] - last[torch.arange(len(rows)), s])


# ---------------------------------------------------------------- graph statistics

def admissible_mask():
    lay = np.arange(N) // H
    return lay[:, None] < lay[None, :]


def threshold(ec_mean, density):
    m = admissible_mask()
    vals = ec_mean[m]
    n_keep = int(round(density * m.sum()))
    cut = np.sort(vals)[::-1][n_keep - 1]
    A = (ec_mean >= cut) & m
    if A.sum() > n_keep:  # break ties deterministically by value order
        order = np.argsort(-ec_mean[m], kind="stable")[:n_keep]
        A = np.zeros_like(m)
        A[tuple(np.argwhere(m)[order].T)] = True
    return A


def rich_club(A):
    deg = A.sum(0) + A.sum(1)
    k = np.percentile(deg, 80)
    R = np.where(deg > k)[0]
    ER = A[np.ix_(R, R)].sum()
    phi = ER / (len(R) * (len(R) - 1)) if len(R) > 1 else np.nan
    return phi, R, k, deg


def null_graph(A, rng, swaps_per_edge=10):
    lay = np.arange(N) // H
    B = A.copy()
    edges = np.argwhere(B)
    groups = {}
    for e in edges:
        groups.setdefault((lay[e[0]], lay[e[1]]), []).append([int(e[0]), int(e[1])])
    for g in groups.values():
        if len(g) < 2:
            continue
        g = [list(e) for e in g]
        for _ in range(swaps_per_edge * len(g)):
            i, j = rng.integers(len(g), size=2)
            if i == j:
                continue
            (a, b), (c, d) = g[i], g[j]
            if b == d or a == c or B[a, d] or B[c, b]:
                continue
            B[a, b] = B[c, d] = False
            B[a, d] = B[c, b] = True
            g[i], g[j] = [a, d], [c, b]
    return B


def louvain_best(A, seeds):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(map(tuple, np.argwhere(A)))
    best, best_q = None, -np.inf
    for s in seeds:
        comms = nx.community.louvain_communities(G, resolution=1, seed=int(s))
        q = nx.community.modularity(G, comms)
        if q > best_q:
            best, best_q = comms, q
    lab = np.empty(N, dtype=int)
    for ci, c in enumerate(best):
        lab[list(c)] = ci
    return lab, best_q


def label_arrays():
    heads, cls = [], []
    for ci, (c, hs) in enumerate(LABELS.items()):
        for h in hs:
            heads.append(idx(h)); cls.append(ci)
    return np.array(heads), np.array(cls)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    smoke = args.smoke
    dev = args.device
    out = Path("results_smoke" if smoke else "results"); out.mkdir(exist_ok=True)
    cfg = dict(n_edge=300, n_held=300, probes=16, nulls=1000, boot=1000, louvain=100, louvain_boot=10,
               perms=1000, null_louvain=200, density=0.10, sens=[0.05, 0.20])
    if smoke:
        cfg.update(n_edge=8, n_held=8, probes=2, nulls=3, boot=2, louvain=2, louvain_boot=1,
                   perms=3, null_louvain=2, sens=[])
    t0 = time.time()
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained_no_processing("gpt2", device=dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    edge_p, seen = make_prompts(model, cfg["n_edge"], SEEDS["edge_prompts"])
    held_p, _ = make_prompts(model, cfg["n_held"], SEEDS["heldout_prompts"], exclude=seen)
    assert not {p["key"] for p in edge_p} & {p["key"] for p in held_p}

    # sanity: manual z equals the model's hook_z
    toks = model.to_tokens([edge_p[0]["clean"]]).to(dev)
    _, cache = cache_run(model, toks)
    zz = z_of_layer(model, 5, cache["blocks.5.hook_resid_pre"])
    err = (zz - cache["blocks.5.attn.hook_z"]).abs().max().item()
    print(f"[check] manual z vs hook_z max abs diff {err:.2e}")
    assert err < 1e-3

    # ---- edges
    gen = torch.Generator().manual_seed(SEEDS["probes"])
    ec_pp = torch.zeros(len(edge_p), N, N)
    for t, rows in by_template(edge_p).items():
        for c0 in range(0, len(rows), 8):
            rr = rows[c0:c0 + 8]
            toks = model.to_tokens([edge_p[i]["clean"] for i in rr]).to(dev)
            ec_pp[rr] = ec_per_prompt(model, toks, cfg["probes"], gen, dev)
    ec_pp = ec_pp.numpy()
    np.save(out / "ec_per_prompt.npy", ec_pp.astype(np.float32))
    ec = ec_pp.mean(0)
    print(f"[edges] {ec_pp.shape} done at {time.time() - t0:.0f}s")

    # ---- held-out runs: clean and corrupt caches, LD, importance by activation patching
    held_groups = by_template(held_p)
    ld_clean = torch.zeros(len(held_p))
    imp = np.zeros(N)
    held_cache = {}
    for t, rows in held_groups.items():
        tc = model.to_tokens([held_p[i]["clean"] for i in rows]).to(dev)
        tx = model.to_tokens([held_p[i]["corrupt"] for i in rows]).to(dev)
        lc, cc = cache_run(model, tc)
        _, cx = cache_run(model, tx)
        held_cache[t] = (rows, cc, cx)
        ld_clean[rows] = logit_diff(model, lc, held_p, rows)
        for h in range(N):
            l, hh = divmod(h, H)
            def patch(z, hook, hh=hh, l=l):
                z[:, :, hh, :] = cx[f"blocks.{l}.attn.hook_z"][:, :, hh, :]
                return z
            with torch.no_grad():
                lp = model.run_with_hooks(tc, fwd_hooks=[(f"blocks.{l}.attn.hook_z", patch)])
            imp[h] += (ld_clean[rows] - logit_diff(model, lp, held_p, rows)).abs().sum().item()
    imp /= len(held_p)
    print(f"[held-out] mean LD clean {ld_clean.mean():.3f}; importance done at {time.time() - t0:.0f}s")

    # ---- edge validation: finite path patching on 20 edges
    A = threshold(ec, cfg["density"])
    E = np.argwhere(A)
    strength = ec[A]
    top = E[np.argsort(-strength)[:10]]
    rest = E[np.argsort(-strength)[10:]]
    rng_v = np.random.default_rng(SEEDS["validation_edges"])
    rnd = rest[rng_v.choice(len(rest), size=min(10, len(rest)), replace=False)]
    val_edges = np.vstack([top, rnd])
    W_O = model.W_O.detach()
    fin = np.zeros((len(val_edges), len(held_p)))
    with torch.no_grad():
        for t, (rows, cc, cx) in held_cache.items():
            for ei, (a, b) in enumerate(val_edges):
                la, ha = divmod(int(a), H); lb, hb = divmod(int(b), H)
                dz = cx[f"blocks.{la}.attn.hook_z"][:, :, ha, :] - cc[f"blocks.{la}.attn.hook_z"][:, :, ha, :]
                r = cc[f"blocks.{lb}.hook_resid_pre"] + dz @ W_O[la, ha]
                zb = z_of_layer(model, lb, r)[:, :, hb, :]
                fin[ei, rows] = (zb - cc[f"blocks.{lb}.attn.hook_z"][:, :, hb, :]).pow(2).sum((1, 2)).cpu().numpy()
    ec_val = ec[val_edges[:, 0], val_edges[:, 1]]
    rho = spearmanr(ec_val, fin.mean(1)).correlation
    rng_b = np.random.default_rng(SEEDS["bootstrap"])
    rho_b = [spearmanr(ec_val, fin[:, rng_b.integers(len(held_p), size=len(held_p))].mean(1)).correlation
             for _ in range(cfg["boot"])]
    print(f"[validation] 20 edges path patched at {time.time() - t0:.0f}s")

    if smoke:
        print("[smoke] shapes ok:", ec_pp.shape, A.sum(), fin.shape, "elapsed", round(time.time() - t0), "s")
        print("[smoke] no test statistics computed or printed")
        return

    # ---- H1 and H2 on the main graph, then sensitivity densities
    hl_heads, hl_cls = label_arrays()
    rng_n = np.random.default_rng(SEEDS["nulls"])
    results = dict(config=cfg, seeds=SEEDS, prereg_sha256=hashlib.sha256(Path("PREREG.md").read_bytes()).hexdigest(),
                   n_edge_prompts=len(edge_p), n_heldout_prompts=len(held_p),
                   mean_ld_clean_heldout=float(ld_clean.mean()), manual_z_check=err)
    for dens in [cfg["density"]] + cfg["sens"]:
        A = threshold(ec, dens)
        phi, R, k, deg = rich_club(A)
        nulls = [null_graph(A, rng_n) for _ in range(cfg["nulls"])]
        assert all((n.sum(0) == A.sum(0)).all() and (n.sum(1) == A.sum(1)).all() for n in nulls[:5])
        phi_null = np.array([rich_club(n)[0] for n in nulls])
        phi_norm = phi / phi_null.mean()
        p_phi = (1 + (phi_null >= phi).sum()) / (1 + len(phi_null))
        I = np.argsort(-imp)[:len(R)]
        ov = len(set(R) & set(I))
        p_ov = hypergeom.sf(ov - 1, N, len(R), len(I))
        lab, q = louvain_best(A, range(SEEDS["louvain_first"], SEEDS["louvain_first"] + cfg["louvain"]))
        ami = adjusted_mutual_info_score(hl_cls, lab[hl_heads])
        rng_p = np.random.default_rng(SEEDS["label_perm"])
        ami_perm = np.array([adjusted_mutual_info_score(rng_p.permutation(hl_cls), lab[hl_heads]) for _ in range(cfg["perms"])])
        p_ami = (1 + (ami_perm >= ami).sum()) / (1 + len(ami_perm))
        ami_nullg = np.array([adjusted_mutual_info_score(hl_cls, louvain_best(n, range(10))[0][hl_heads])
                              for n in nulls[:cfg["null_louvain"]]])
        res = dict(density=dens, n_edges=int(A.sum()), k=float(k), club=[name(i) for i in R],
                   phi=float(phi), phi_null_mean=float(phi_null.mean()), phi_null_p975=float(np.percentile(phi_null, 97.5)),
                   phi_norm=float(phi_norm), p_phi=float(p_phi),
                   importance_top=[name(i) for i in I], overlap=ov, p_overlap=float(p_ov),
                   modularity_Q=float(q), n_communities=int(lab.max() + 1), ami=float(ami),
                   ami_perm_p975=float(np.percentile(ami_perm, 97.5)), p_ami=float(p_ami),
                   ami_nullgraph_mean=float(ami_nullg.mean()), ami_nullgraph_p975=float(np.percentile(ami_nullg, 97.5)),
                   communities_of_labelled={name(h): int(lab[h]) for h in hl_heads})
        res["H1_fail_a"] = bool(phi_norm <= 1 or p_phi >= 0.025)
        res["H1_fail_b"] = bool(p_ov >= 0.025)
        res["H1_verdict"] = "fail" if (res["H1_fail_a"] or res["H1_fail_b"]) else "pass"
        res["H2_verdict"] = "fail" if p_ami >= 0.025 else "pass"
        if dens == cfg["density"]:
            # bootstrap over edge prompts (null mean fixed from main run)
            rb = np.random.default_rng(SEEDS["bootstrap"])
            bphi, bov, bami = [], [], []
            for _ in range(cfg["boot"]):
                ecb = ec_pp[rb.integers(len(edge_p), size=len(edge_p))].mean(0)
                Ab = threshold(ecb, dens)
                pb, Rb, _, _ = rich_club(Ab)
                bphi.append(pb / phi_null.mean())
                bov.append(len(set(Rb) & set(np.argsort(-imp)[:len(Rb)])))
                lb_, _ = louvain_best(Ab, range(cfg["louvain_boot"]))
                bami.append(adjusted_mutual_info_score(hl_cls, lb_[hl_heads]))
            res["phi_norm_ci95"] = [float(np.percentile(bphi, 2.5)), float(np.percentile(bphi, 97.5))]
            res["overlap_ci95"] = [float(np.percentile(bov, 2.5)), float(np.percentile(bov, 97.5))]
            res["ami_ci95"] = [float(np.percentile(bami, 2.5)), float(np.percentile(bami, 97.5))]
            results["main"] = res
        else:
            results.setdefault("sensitivity", []).append(res)
        print(f"[density {dens}] done at {time.time() - t0:.0f}s")
    results["validation"] = dict(edges=[[name(a), name(b)] for a, b in val_edges],
                                 ec=ec_val.tolist(), finite_effect=fin.mean(1).tolist(),
                                 spearman=float(rho), spearman_ci95=[float(np.nanpercentile(rho_b, 2.5)), float(np.nanpercentile(rho_b, 97.5))])
    results["importance"] = {name(i): float(imp[i]) for i in range(N)}
    results["elapsed_s"] = round(time.time() - t0)
    np.save(out / "ec_mean.npy", ec.astype(np.float32))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    m = results["main"]
    print(json.dumps({k: m[k] for k in ["phi_norm", "phi_norm_ci95", "p_phi", "overlap", "p_overlap", "H1_verdict",
                                        "ami", "ami_ci95", "p_ami", "H2_verdict"]}, indent=1))
    print("validation spearman", results["validation"]["spearman"], results["validation"]["spearman_ci95"])


if __name__ == "__main__":
    main()
