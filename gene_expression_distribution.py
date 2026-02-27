#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib
mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42   

# -------------------- Config --------------------
ALPHA_AD = 0.05
DELTA_BIC_STRONG = 10.0
MIN_WEIGHT_2G = 0.10
SEP_SIGMA_2G = 1.0
MAX_EXPAND = 4000
MAX_PLOT_PER_CLASS = 60
VIEW = (24, -58)
BOX_ASPECT = (1.2, 3, 1.0)
RANDOM_SEED = 0

# -------------------- IO --------------------
def load_histogram(csv_path):
    df = pd.read_csv(csv_path)
    centers = df.iloc[:, 0].astype(float).values
    genes = list(df.columns[1:])
    C = df.iloc[:, 1:].values.astype(float)
    return centers, genes, C

def build_edges_from_centers(centers):
    mid = (centers[:-1] + centers[1:]) / 2.0
    edges = np.concatenate(([centers[0] - (mid[0] - centers[0])], mid,
                            [centers[-1] + (centers[-1] - mid[-1])]))
    return edges

def expand_values(values, freq, max_n=MAX_EXPAND, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    r = np.rint(np.asarray(freq)).astype(int)
    dat = np.repeat(values, r)
    if len(dat) > max_n:
        idx = rng.choice(len(dat), size=max_n, replace=False)
        dat = dat[idx]
    return dat.astype(float)

# -------------------- Stats --------------------
def bic_from_loglik(ll, k, n):
    return k * np.log(n) - 2.0 * ll

def ad_test_normal(dat):
    if len(dat) < 8:
        return np.nan, np.nan, False
    res = stats.anderson(dat, dist='norm')
    crit5 = None
    for lev, crit in zip(res.significance_level, res.critical_values):
        if abs(lev - 5.0) < 1e-6:
            crit5 = crit
            break
    stat = float(res.statistic)
    reject = (stat > crit5) if (crit5 is not None) else False
    return stat, float(crit5) if crit5 is not None else np.nan, reject

def fit_skewnorm_log2(dat):
    try:
        a, loc, scale = stats.skewnorm.fit(dat)
        ll = float(np.sum(stats.skewnorm.logpdf(dat, a, loc=loc, scale=scale)))
        return ll, 3, {"a": a, "loc": loc, "scale": scale}
    except Exception:
        return np.nan, 3, None

def fit_t_log2(dat):
    try:
        df, loc, scale = stats.t.fit(dat)
        ll = float(np.sum(stats.t.logpdf(dat, df, loc=loc, scale=scale)))
        return ll, 3, {"df": df, "loc": loc, "scale": scale}
    except Exception:
        return np.nan, 3, None

def fit_lognorm_linear(y):
    try:
        shape, loc, scale = stats.lognorm.fit(y, floc=0)
        ll = float(np.sum(stats.lognorm.logpdf(y, s=shape, loc=loc, scale=scale)))
        return ll, 2, {"shape": shape, "loc": loc, "scale": scale}
    except Exception:
        return np.nan, 2, None

def fit_gamma_linear(y):
    try:
        a, loc, scale = stats.gamma.fit(y, floc=0)
        ll = float(np.sum(stats.gamma.logpdf(y, a, loc=loc, scale=scale)))
        return ll, 2, {"shape": a, "loc": loc, "scale": scale}
    except Exception:
        return np.nan, 2, None

def fit_gmm(dat_log2, n_components):
    X = dat_log2.reshape(-1, 1).astype(float)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=RANDOM_SEED).fit(X)
    bic = gmm.bic(X)  
    ll = float(np.sum(gmm.score_samples(X)))
    return gmm, bic, ll

def gmm_2g_subtype(weights, means, sigmas):
    w = np.array(weights).flatten()
    m = np.array(means).flatten()
    s = np.array(sigmas).flatten()
    order = np.argsort(m)
    w, m, s = w[order], m[order], s[order]
    pooled_sigma = np.sqrt((s[0]**2 + s[1]**2) / 2.0)
    mean_sep = abs(m[1] - m[0])
    weight_ok = (w.min() >= MIN_WEIGHT_2G)
    sep_ok = (mean_sep >= SEP_SIGMA_2G * pooled_sigma)
    return ("2G_true" if (weight_ok and sep_ok) else "2G_pseudo"), (w, m, s), pooled_sigma, mean_sep

def pdf_normal_log2(x, mu, sigma):
    return stats.norm.pdf(x, loc=mu, scale=sigma)

def pdf_skewnormal_log2(x, a, loc, scale):
    return stats.skewnorm.pdf(x, a, loc=loc, scale=scale)

def pdf_t_log2(x, df, loc, scale):
    return stats.t.pdf(x, df, loc=loc, scale=scale)

def pdf_lognormal_as_log2(x, shape, loc, scale):
    y = np.power(2.0, x)          
    return stats.lognorm.pdf(y, s=shape, loc=loc, scale=scale) * (np.log(2.0) * y)

def pdf_gamma_as_log2(x, shape, loc, scale):
    y = np.power(2.0, x)
    return stats.gamma.pdf(y, a=shape, loc=loc, scale=scale) * (np.log(2.0) * y)

# -------------------- Plotting --------------------
def plot_mountain_with_overlays(x, C_sel, names_sel, outfile, title,
                                overlay_funcs=None, overlay_params=None,
                                boundary_margin=0.8, gap=1.0, 
                                colormap='plasma', fit_only=False):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    nbins, nsel = C_sel.shape
    if nsel == 0:
        return    
    peak_values = []
    for i in range(nsel):
        if overlay_funcs is not None and overlay_params is not None:
            f = overlay_funcs[i]
            p = overlay_params[i]
            if f is not None and p is not None:
                y_vals = f(x, **p)
                peak_val = np.max(y_vals)
            else:
                peak_val = 0
        else:
            peak_val = 0
        peak_values.append(peak_val)
    sorted_indices = np.argsort(peak_values)
    C_sel = C_sel[:, sorted_indices]
    names_sel = [names_sel[i] for i in sorted_indices]
    if overlay_funcs is not None and overlay_params is not None:
        overlay_funcs = [overlay_funcs[i] for i in sorted_indices]
        overlay_params = [overlay_params[i] for i in sorted_indices]
    if nsel == 1:
        y_positions = np.array([0.0])
        y_min, y_max = -1.0, 1.0
    else:
        available_space = (nsel - 1)  
        total_range = available_space * gap
        y_min = -boundary_margin
        y_max = total_range + boundary_margin
        y_positions = np.linspace(0, total_range, nsel)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    cmap = matplotlib.colormaps.get_cmap(colormap)
    colors = [cmap(i / max(1, nsel - 1)) for i in range(nsel)]

    for i in range(nsel):
        y = np.full_like(x, y_positions[i])
        if not fit_only:
            c = C_sel[:, i]
            if c.sum() <= 0:
                continue
            prob = (c / c.sum())  
            ax.plot(x, y, prob, linewidth=1.6, alpha=0.95)
        
        if overlay_funcs is not None and overlay_params is not None:
            f = overlay_funcs[i]
            p = overlay_params[i]
            if f is not None and p is not None:
                from scipy.integrate import quad
                edges = build_edges_from_centers(x)
                widths = np.diff(edges)
                pmf_vals = np.array([quad(lambda xx: f(xx, **p), edges[j], edges[j+1])[0]
                                     for j in range(len(widths))])
                mdl = pmf_vals
                ax.plot(x, y, mdl, linestyle="-", linewidth=2.0, alpha=0.9, color=colors[i])

    ax.set_xlabel("log2(Expression)", fontsize=12)
    ax.set_ylabel("Gene (depth)", fontsize=12)
    ax.set_zlabel("Probability per bin (across samples)", fontsize=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names_sel, fontsize=7)
    ax.view_init(elev=VIEW[0], azim=VIEW[1])
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0, None)
    try:
        ax.set_box_aspect(BOX_ASPECT)
    except Exception:
        pass
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, format='pdf', bbox_inches='tight',
                pad_inches=0.1, transparent=False)
    plt.close(fig)

# -------------------- Main analysis --------------------
def analyze_species(csv_path, species_name, args):
    x, genes, C = load_histogram(csv_path)
    nbins, ngen = C.shape

    results = []
    for j, g in enumerate(genes):
        dat = expand_values(x, C[:, j], seed=RANDOM_SEED)
        nobs = len(dat)
        if nobs < 8:
            results.append({"gene": g, "species": species_name, "category": "insufficient"})
            continue

        ad_stat, ad_crit5, ad_reject = ad_test_normal(dat)
        if not ad_reject:
            mu = float(np.mean(dat)); sg = float(np.std(dat, ddof=0))
            results.append({"gene": g, "species": species_name, "category": "Normal",
                            "mu": mu, "sigma": sg,
                            "AD_stat": ad_stat, "AD_crit_5pct": ad_crit5, "AD_reject": False})
            continue

        y_lin = np.power(2.0, dat)
        ll_skew, k_skew, p_skew = fit_skewnorm_log2(dat)
        bic_skew = bic_from_loglik(ll_skew, k_skew, len(dat)) if np.isfinite(ll_skew) else np.nan

        ll_t, k_t, p_t = fit_t_log2(dat)
        bic_t = bic_from_loglik(ll_t, k_t, len(dat)) if np.isfinite(ll_t) else np.nan

        ll_logn, k_logn, p_logn = fit_lognorm_linear(y_lin)
        bic_logn = bic_from_loglik(ll_logn, k_logn, len(y_lin)) if np.isfinite(ll_logn) else np.nan

        ll_gam, k_gam, p_gam = fit_gamma_linear(y_lin)
        bic_gam = bic_from_loglik(ll_gam, k_gam, len(y_lin)) if np.isfinite(ll_gam) else np.nan

        try:
            gmm2, bic_2g, ll_2g = fit_gmm(dat, 2)
            w = gmm2.weights_.flatten()
            m = gmm2.means_.flatten()
            s = np.sqrt(gmm2.covariances_.flatten())
            subtype, (w_s, m_s, s_s), pooled_sigma, mean_sep = gmm_2g_subtype(w, m, s)
        except Exception:
            bic_2g = np.nan
            subtype = None
            w_s = m_s = s_s = pooled_sigma = mean_sep = np.nan

        family_bic = {
            "single-skewnormal_log2": bic_skew,
            "single-t_log2": bic_t,
            "single-lognormal_linear": bic_logn,
            "single-gamma_linear": bic_gam,
            "2G_mixture_log2": bic_2g
        }
        valid = {k: v for k, v in family_bic.items() if pd.notnull(v) and np.isfinite(v)}
        if len(valid) == 0:
            results.append({"gene": g, "species": species_name, "category": "fit_error",
                            "AD_stat": ad_stat, "AD_crit_5pct": ad_crit5, "AD_reject": True})
            continue

        best_family = min(valid, key=valid.get)
        rec = {"gene": g, "species": species_name, "category": best_family,
               "AD_stat": ad_stat, "AD_crit_5pct": ad_crit5, "AD_reject": True,
               "BIC_skewnormal": bic_skew, "BIC_t": bic_t, "BIC_lognormal": bic_logn, "BIC_gamma": bic_gam,
               "BIC_2G": bic_2g}

        if best_family == "2G_mixture_log2":
            rec.update({
                "subtype": subtype,
                "w1": w_s[0], "w2": w_s[1],
                "m1": m_s[0], "m2": m_s[1],
                "s1": s_s[0], "s2": s_s[1],
                "pooled_sigma": pooled_sigma, "mean_sep": mean_sep
            })
        elif best_family == "single-skewnormal_log2" and p_skew is not None:
            rec.update({"a": p_skew["a"], "loc": p_skew["loc"], "scale": p_skew["scale"]})
        elif best_family == "single-t_log2" and p_t is not None:
            rec.update({"df": p_t["df"], "loc": p_t["loc"], "scale": p_t["scale"]})
        elif best_family == "single-lognormal_linear" and p_logn is not None:
            rec.update({"shape": p_logn["shape"], "loc": p_logn["loc"], "scale": p_logn["scale"]})
        elif best_family == "single-gamma_linear" and p_gam is not None:
            rec.update({"shape": p_gam["shape"], "loc": p_gam["loc"], "scale": p_gam["scale"]})
        results.append(rec)

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{species_name}_unified_classification.csv", index=False)

    cat_for_count = []
    for _, row in res_df.iterrows():
        if row["category"] == "2G_mixture_log2":
            cat_for_count.append(row.get("subtype", "2G_mixture_log2"))
        else:
            cat_for_count.append(row["category"])
    counts = pd.Series(cat_for_count).value_counts(dropna=False)
    counts.to_csv(f"{species_name}_unified_category_counts.csv")

    edges = build_edges_from_centers(x)
    widths = np.diff(edges)

    def select_and_plot(cat_key, overlay_builder, pretty_name=None, colormap='plasma'):
        idx = []
        for i, row in res_df.iterrows():
            if cat_key in ("2G_true", "2G_pseudo"):
                if row["category"] == "2G_mixture_log2" and row.get("subtype", None) == cat_key:
                    idx.append(i)
            else:
                if row["category"] == cat_key:
                    idx.append(i)
        if len(idx) == 0:
            print(f"No genes found for category: {cat_key}")
            return
        if len(idx) > MAX_PLOT_PER_CLASS:
            idx = idx[:MAX_PLOT_PER_CLASS]

        names_sel = [genes[i] for i in idx]
        C_sel = C[:, idx]

        overlay_funcs, overlay_params = [], []
        for i in idx:
            params = res_df.iloc[i].to_dict()
            f, p = overlay_builder(params)
            overlay_funcs.append(f)
            overlay_params.append(p)

        outname = pretty_name if pretty_name else cat_key
        outfile = f"{species_name}_3d_{outname}.pdf"
        title = f"{species_name} | {outname} (n={len(idx)})"
        print(f"Plotting {len(idx)} genes for {outname}...")
        plot_mountain_with_overlays(x, C_sel, names_sel, outfile, title,
                                    overlay_funcs=overlay_funcs, overlay_params=overlay_params,
                                    boundary_margin=args.boundary_margin, gap=args.gap,
                                    colormap=colormap, fit_only=args.fit_only)

    def plot_combined_2g():
        true_idx = []
        for i, row in res_df.iterrows():
            if row["category"] == "2G_mixture_log2" and row.get("subtype", None) == "2G_true":
                true_idx.append(i)
        
        pseudo_idx = []
        for i, row in res_df.iterrows():
            if row["category"] == "2G_mixture_log2" and row.get("subtype", None) == "2G_pseudo":
                pseudo_idx.append(i)
        
        if len(true_idx) > MAX_PLOT_PER_CLASS:
            true_idx = true_idx[:MAX_PLOT_PER_CLASS]
        if len(pseudo_idx) > MAX_PLOT_PER_CLASS:
            pseudo_idx = pseudo_idx[:MAX_PLOT_PER_CLASS]
        ）
        combined_idx = true_idx + pseudo_idx
        
        if len(combined_idx) == 0:
            print("No genes found for combined 2G_true and 2G_pseudo")
            return
        
        names_sel = [genes[i] for i in combined_idx]
        C_sel = C[:, combined_idx]
        
        overlay_funcs, overlay_params = [], []
        for i in combined_idx:
            params = res_df.iloc[i].to_dict()
            f, p = ov_2g(params)
            overlay_funcs.append(f)
            overlay_params.append(p)
        
        peak_values = []
        for i in range(len(combined_idx)):
            if overlay_funcs[i] is not None and overlay_params[i] is not None:
                f = overlay_funcs[i]
                p = overlay_params[i]
                y_vals = f(x, **p)
                peak_val = np.max(y_vals)
            else:
                peak_val = 0
            peak_values.append((i, peak_val))
        
        peak_values.sort(key=lambda x: x[1])
        
        sorted_indices = [item[0] for item in peak_values]
        
        C_sel = C_sel[:, sorted_indices]
        names_sel = [names_sel[i] for i in sorted_indices]
        overlay_funcs = [overlay_funcs[i] for i in sorted_indices]
        overlay_params = [overlay_params[i] for i in sorted_indices]
        
        nsel = len(combined_idx)
        if nsel == 1:
            y_positions = np.array([0.0])
            y_min, y_max = -1.0, 1.0
        else:
            boundary_margin = args.boundary_margin
            available_space = (nsel - 1)
            gap = args.gap
            total_range = available_space * gap
            y_min = -boundary_margin
            y_max = total_range + boundary_margin
            y_positions = np.linspace(0, total_range, nsel)
        
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")
        
        cmap = matplotlib.colormaps.get_cmap(args.combined_colormap)
        colors = [cmap(i / max(1, nsel - 1)) for i in range(nsel)]
        
        for i in range(nsel):
            y = np.full_like(x, y_positions[i])
            
            if not args.fit_only:
                c = C_sel[:, i]
                if c.sum() <= 0:
                    continue
                prob = (c / c.sum())  # per-bin probability (PMF)
                ax.plot(x, y, prob, linewidth=1.6, alpha=0.95)
            
            if overlay_funcs[i] is not None and overlay_params[i] is not None:
                from scipy.integrate import quad
                edges = build_edges_from_centers(x)
                widths = np.diff(edges)
                f = overlay_funcs[i]
                p = overlay_params[i]
                pmf_vals = np.array([quad(lambda xx: f(xx, **p), edges[j], edges[j+1])[0]
                                     for j in range(len(widths))])
                mdl = pmf_vals
        
                ax.plot(x, y, mdl, linestyle="-", linewidth=2.0, alpha=0.9, color=colors[i])
        
        ax.set_xlabel("log2(Expression)", fontsize=12)
        ax.set_ylabel("Gene (depth)", fontsize=12)
        ax.set_zlabel("Probability per bin (across samples)", fontsize=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names_sel, fontsize=7)
        ax.view_init(elev=VIEW[0], azim=VIEW[1])
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0, None)
        
        try:
            ax.set_box_aspect(BOX_ASPECT)
        except Exception:
            pass
        
        plt.title(f"{species_name} | 2G_true + 2G_pseudo (n={nsel})", fontsize=14)
        plt.tight_layout()
        
        outfile = f"{species_name}_3d_combined_2g.pdf"
        plt.savefig(outfile, dpi=300, format='pdf', bbox_inches='tight',
                    pad_inches=0.1, transparent=False)
        plt.close(fig)
        
        print(f"Plotting combined 2G_true and 2G_pseudo (n={nsel})...")

    def plot_combined_normal_skew_t():
        normal_idx = []
        skew_idx = []
        t_idx = []
        
        for i, row in res_df.iterrows():
            if row["category"] == "Normal":
                normal_idx.append(i)
            elif row["category"] == "single-skewnormal_log2":
                skew_idx.append(i)
            elif row["category"] == "single-t_log2":
                t_idx.append(i)
        
        if len(normal_idx) > MAX_PLOT_PER_CLASS:
            normal_idx = normal_idx[:MAX_PLOT_PER_CLASS]
        if len(skew_idx) > MAX_PLOT_PER_CLASS:
            skew_idx = skew_idx[:MAX_PLOT_PER_CLASS]
        if len(t_idx) > MAX_PLOT_PER_CLASS:
            t_idx = t_idx[:MAX_PLOT_PER_CLASS]
        
        combined_idx = normal_idx + skew_idx + t_idx
        
        if len(combined_idx) == 0:
            print("No genes found for combined Normal, single-skewnormal_log2 and single-t_log2")
            return
        
        names_sel = [genes[i] for i in combined_idx]
        C_sel = C[:, combined_idx]
        
        overlay_funcs, overlay_params = [], []
        for i in combined_idx:
            params = res_df.iloc[i].to_dict()
            if params["category"] == "Normal":
                f, p = ov_normal(params)
            elif params["category"] == "single-skewnormal_log2":
                f, p = ov_skew(params)
            else:  # single-t_log2
                f, p = ov_t(params)
            overlay_funcs.append(f)
            overlay_params.append(p)
        
        peak_values = []
        for i in range(len(combined_idx)):
            if overlay_funcs[i] is not None and overlay_params[i] is not None:
                f = overlay_funcs[i]
                p = overlay_params[i]
                y_vals = f(x, **p)
                peak_val = np.max(y_vals)
            else:
                peak_val = 0
            peak_values.append((i, peak_val))
        
        peak_values.sort(key=lambda x: x[1])
        
        sorted_indices = [item[0] for item in peak_values]
        C_sel = C_sel[:, sorted_indices]
        names_sel = [names_sel[i] for i in sorted_indices]
        overlay_funcs = [overlay_funcs[i] for i in sorted_indices]
        overlay_params = [overlay_params[i] for i in sorted_indices]
        
        nsel = len(combined_idx)
        if nsel == 1:
            y_positions = np.array([0.0])
            y_min, y_max = -1.0, 1.0
        else:
            boundary_margin = args.boundary_margin
            available_space = (nsel - 1)
            gap = args.gap
            total_range = available_space * gap
            y_min = -boundary_margin
            y_max = total_range + boundary_margin
            y_positions = np.linspace(0, total_range, nsel)
        
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection="3d")
        
        cmap = matplotlib.colormaps.get_cmap(args.combined_colormap)
        colors = [cmap(i / max(1, nsel - 1)) for i in range(nsel)]
        
        for i in range(nsel):
            y = np.full_like(x, y_positions[i])
            if not args.fit_only:
                c = C_sel[:, i]
                if c.sum() <= 0:
                    continue
                prob = (c / c.sum())  
                ax.plot(x, y, prob, linewidth=1.6, alpha=0.95)
            if overlay_funcs[i] is not None and overlay_params[i] is not None:
                from scipy.integrate import quad
                edges = build_edges_from_centers(x)
                widths = np.diff(edges)
                f = overlay_funcs[i]
                p = overlay_params[i]
                pmf_vals = np.array([quad(lambda xx: f(xx, **p), edges[j], edges[j+1])[0]
                                     for j in range(len(widths))])
                mdl = pmf_vals
                ax.plot(x, y, mdl, linestyle="-", linewidth=2.0, alpha=0.9, color=colors[i])
        
        ax.set_xlabel("log2(Expression)", fontsize=12)
        ax.set_ylabel("Gene (depth)", fontsize=12)
        ax.set_zlabel("Probability per bin (across samples)", fontsize=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names_sel, fontsize=7)
        ax.view_init(elev=VIEW[0], azim=VIEW[1])
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0, None)
        
        try:
            ax.set_box_aspect(BOX_ASPECT)
        except Exception:
            pass
        
        plt.title(f"{species_name} | Normal + single-skewnormal_log2 + single-t_log2 (n={nsel})", fontsize=14)
        plt.tight_layout()
        
        # save pdf
        outfile = f"{species_name}_3d_combined_normal_skew_t.pdf"
        plt.savefig(outfile, dpi=300, format='pdf', bbox_inches='tight',
                    pad_inches=0.1, transparent=False)
        plt.close(fig)
        
        print(f"Plotting combined Normal, single-skewnormal_log2 and single-t_log2 (n={nsel})...")

    # Overlay builders
    def ov_normal(params):
        if ("mu" in params) and ("sigma" in params) and params["sigma"] and params["sigma"]>0:
            return (lambda x, mu, sigma: stats.norm.pdf(x, loc=mu, scale=sigma)), {"mu": params["mu"], "sigma": params["sigma"]}
        return None, None

    def ov_2g(params):
        if all(k in params for k in ["w1","w2","m1","m2","s1","s2"]):
            def f(x, w1, w2, m1, m2, s1, s2):
                return w1*stats.norm.pdf(x, loc=m1, scale=s1) + w2*stats.norm.pdf(x, loc=m2, scale=s2)
            return f, {"w1":params["w1"], "w2":params["w2"], "m1":params["m1"], "m2":params["m2"], "s1":params["s1"], "s2":params["s2"]}
        return None, None

    def ov_skew(params):
        if all(k in params for k in ["a","loc","scale"]):
            return (lambda x, a, loc, scale: stats.skewnorm.pdf(x, a, loc=loc, scale=scale)), \
                   {"a":params["a"], "loc":params["loc"], "scale":params["scale"]}
        return None, None

    def ov_t(params):
        if all(k in params for k in ["df","loc","scale"]):
             # Visualization clamp: avoid near-flat or ultra-spiky overlays
             df = params["df"]
             loc = params["loc"]
             scale = min(max(params["scale"], 0.05), 5.0)
             return (lambda x, df, loc, scale: stats.t.pdf(x, df, loc=loc, scale=scale)), \
                    {"df": df, "loc": loc, "scale": scale}
        return None, None

    def ov_lognorm(params):
        if all(k in params for k in ["shape","loc","scale"]):
            def f(x, shape, loc, scale):
                y = np.power(2.0, x)
                return stats.lognorm.pdf(y, s=shape, loc=loc, scale=scale) * (np.log(2.0) * y)
            return f, {"shape":params["shape"], "loc":params["loc"], "scale":params["scale"]}
        return None, None

    def ov_gamma(params):
        if all(k in params for k in ["shape","loc","scale"]):
            def f(x, shape, loc, scale):
                y = np.power(2.0, x)
                return stats.gamma.pdf(y, a=shape, loc=loc, scale=scale) * (np.log(2.0) * y)
            return f, {"shape":params["shape"], "loc":params["loc"], "scale":params["scale"]}
        return None, None

    plot_combined_2g()
    plot_combined_normal_skew_t()
    select_and_plot("single-lognormal_linear", ov_lognorm, pretty_name="single-lognormal_linear", 
                   colormap=args.combined_colormap)
    select_and_plot("single-gamma_linear", ov_gamma, pretty_name="single-gamma_linear", 
                   colormap=args.combined_colormap)

    print(f"[{species_name}] Wrote classification & PDF plots.")
    return res_df

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Unified model selection: Normal vs Single-peak vs 2G mixture.")
    parser.add_argument("--steinii", type=str, required=True, help="Steinii across-samples histogram CSV (log2 centers + counts)")
    parser.add_argument("--inflata", type=str, required=True, help="Inflata across-samples histogram CSV (log2 centers + counts)")
    parser.add_argument("--boundary_margin", type=float, default=0.8, 
                       help="Distance from axis to first/last curve (default: 0.8)")
    parser.add_argument("--gap", type=float, default=1.0, 
                       help="Base gap between curves (default: 1.0)")
    parser.add_argument("--fit_only", action="store_true", 
                       help="Only show fitted curves (no histogram data)")
    parser.add_argument("--combined_colormap", type=str, default="plasma", 
                       help="Colormap for all combined plots (default: plasma)")
    
    args = parser.parse_args()

    analyze_species(args.steinii, species_name="steinii", args=args)
    analyze_species(args.inflata, species_name="inflata", args=args)

if __name__ == "__main__":
    main()
