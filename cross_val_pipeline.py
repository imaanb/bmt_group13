"""
Wessels et al. (2005) double cross-validation pipeline with Relief-F feature selection.

Protocol:
  - Outer loop : 3-fold CV, repeated 100 times  (validation loop)
  - Inner loop : 10-fold CV                      (training / parameter optimisation loop)
  - Performance metric: average of sensitivity & specificity  ( (TP/P + TN/N) / 2 )
  - Feature selector: Relief-F, k features evaluated from k_candidates
  - Models: Random Forest, KNN, SVM
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from ReliefF import ReliefF
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# ─────────────────────────────────────────────
# 1.  Helpers
# ─────────────────────────────────────────────





def avg_sens_spec(y_true, y_pred):
    """Average one-vs-rest sensitivity/specificity across observed classes."""
    class_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    per_class_scores = []

    for label in class_labels:
        y_true_bin = np.asarray(y_true) == label
        y_pred_bin = np.asarray(y_pred) == label
        tn, fp, fn, tp = confusion_matrix(
            y_true_bin,
            y_pred_bin,
            labels=[False, True]
        ).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_scores.append((sens + spec) / 2.0)

    return float(np.mean(per_class_scores))


def per_class_sens_spec(y_true, y_pred):
    """Per-class one-vs-rest (sensitivity + specificity) / 2."""
    class_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    scores = {}
    for label in class_labels:
        y_true_bin = np.asarray(y_true) == label
        y_pred_bin = np.asarray(y_pred) == label
        tn, fp, fn, tp = confusion_matrix(
            y_true_bin, y_pred_bin, labels=[False, True]
        ).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        scores[label] = (sens + spec) / 2.0
    return scores


def relief_f_top_k(X, y, k):
    """Return indices of top-k features ranked by Relief-F."""
    n_neighbors = min(10, len(y) - 1)
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=k)
    fs.fit(X, y)
    return fs.top_features[:k]


def get_model(name, **kwargs):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=kwargs.get("knn_k", 5))
    elif name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    
    raise ValueError(f"Unknown model: {name}")


def save_best_val_features(all_results, X, y, feature_info, output_dir="output"):
    """Save the selected features for the validation-optimal k (overall and per-class),
    including per-fold frequency.

    Parameters
    ----------
    feature_info : pd.DataFrame
        Must contain columns 'Chromosome', 'Start', 'End' (one row per feature, same order as X columns).
    """
    os.makedirs(output_dir, exist_ok=True)

    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    for mname, res in all_results.items():
        best_k_val = max(res["k_inner_val"], key=res["k_inner_val"].get)

        # ── Per-fold frequency table ──────────────────────────────
        n_features = X.shape[1]
        freq = np.zeros(n_features, dtype=int)
        for sel in res["fold_features"]:
            freq[sel] += 1
        n_folds = len(res["fold_features"])

        freq_df = pd.DataFrame({
            "feature_index": np.arange(n_features),
            "Chromosome": feature_info["Chromosome"].values,
            "Start": feature_info["Start"].values,
            "End": feature_info["End"].values,
            "selection_count": freq,
            "selection_freq": freq / n_folds,
        })
        freq_df = freq_df[freq_df["selection_count"] > 0]
        freq_df = freq_df.sort_values("selection_count", ascending=False).reset_index(drop=True)

        freq_path = os.path.join(output_dir, f"{mname.lower()}_feature_selection_frequency.csv")
        freq_df.to_csv(freq_path, index=False)
        print(f"Feature frequency saved -> {freq_path}")

        # ── Canonical feature set (overall best_k_val) ───────────
        k_safe = min(best_k_val, X_sc.shape[1])
        sel_idx = relief_f_top_k(X_sc, y, k_safe)

        best_df = pd.DataFrame({
            "feature_rank": np.arange(1, len(sel_idx) + 1),
            "feature_index": sel_idx,
            "Chromosome": feature_info["Chromosome"].values[sel_idx],
            "Start": feature_info["Start"].values[sel_idx],
            "End": feature_info["End"].values[sel_idx],
        })
        best_path = os.path.join(output_dir, f"{mname.lower()}_best_k_val_features.csv")
        best_df.to_csv(best_path, index=False)
        print(f"Best k(val)={best_k_val} features saved -> {best_path}")

        # ── Per-class optimal k feature sets ─────────────────────
        if "k_inner_val_pc" in res:
            for cls in sorted(res["k_inner_val_pc"].keys()):
                best_k_cls = max(res["k_inner_val_pc"][cls],
                                 key=res["k_inner_val_pc"][cls].get)
                k_safe_cls = min(best_k_cls, X_sc.shape[1])
                sel_idx_cls = relief_f_top_k(X_sc, y, k_safe_cls)

                cls_df = pd.DataFrame({
                    "feature_rank": np.arange(1, len(sel_idx_cls) + 1),
                    "feature_index": sel_idx_cls,
                    "Chromosome": feature_info["Chromosome"].values[sel_idx_cls],
                    "Start": feature_info["Start"].values[sel_idx_cls],
                    "End": feature_info["End"].values[sel_idx_cls],
                })
                cls_safe = str(cls).replace(" ", "_").replace("/", "-")
                cls_path = os.path.join(output_dir,
                                        f"{mname.lower()}_best_k_val_features_class_{cls_safe}.csv")
                cls_df.to_csv(cls_path, index=False)
                print(f"  Class '{cls}' best k(val)={best_k_cls} features saved -> {cls_path}")


# ─────────────────────────────────────────────
# 2.  Inner loop — optimize k 
# ─────────────────────────────────────────────

def inner_loop(X_train, y_train, model_name, k_candidates, n_inner_folds=10, n_inner_reps=5):
    """
    10-fold CV (repeated 5×) on the training set to find the best k.
    Returns: best_k, dict{k: mean_train_perf}, dict{k: mean_val_perf},
             dict{class: {k: mean_train_perf}}, dict{class: {k: mean_val_perf}}
    """
    classes = np.unique(y_train)
    k_train = {k: [] for k in k_candidates}
    k_val   = {k: [] for k in k_candidates}
    k_train_pc = {cls: {k: [] for k in k_candidates} for cls in classes}
    k_val_pc   = {cls: {k: [] for k in k_candidates} for cls in classes}

    for _ in range(n_inner_reps):
        skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=None)
        for tr_idx, te_idx in skf.split(X_train, y_train):
            X_tr, X_te = X_train[tr_idx], X_train[te_idx]
            y_tr, y_te = y_train[tr_idx], y_train[te_idx]

            # scale inside fold to avoid leakage
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            for k in k_candidates:
                k_safe = min(k, X_tr.shape[1])
                sel_idx = relief_f_top_k(X_tr, y_tr, k_safe)
                X_tr_k = X_tr[:, sel_idx]
                X_te_k = X_te[:, sel_idx]

                model = get_model(model_name)
                model.fit(X_tr_k, y_tr)

                y_pred_tr = model.predict(X_tr_k)
                y_pred_te = model.predict(X_te_k)

                k_train[k].append(avg_sens_spec(y_tr, y_pred_tr))
                k_val[k].append(avg_sens_spec(y_te, y_pred_te))

                pc_tr = per_class_sens_spec(y_tr, y_pred_tr)
                pc_te = per_class_sens_spec(y_te, y_pred_te)
                for cls in classes:
                    if cls in pc_tr:
                        k_train_pc[cls][k].append(pc_tr[cls])
                    if cls in pc_te:
                        k_val_pc[cls][k].append(pc_te[cls])

    mean_val = {k: np.mean(k_val[k]) for k in k_candidates}
    best_k   = max(mean_val, key=mean_val.get)

    mean_train_pc = {cls: {k: np.mean(k_train_pc[cls][k]) if k_train_pc[cls][k] else 0.0
                           for k in k_candidates} for cls in classes}
    mean_val_pc   = {cls: {k: np.mean(k_val_pc[cls][k]) if k_val_pc[cls][k] else 0.0
                           for k in k_candidates} for cls in classes}

    return (best_k,
            {k: np.mean(k_train[k]) for k in k_candidates},
            mean_val,
            mean_train_pc,
            mean_val_pc)


# ─────────────────────────────────────────────
# 3.  Outer loop 
# ─────────────────────────────────────────────

def wessels_pipeline(X, y, model_name, k_candidates,
                     n_outer_folds=3, n_outer_reps=100,
                     n_inner_folds=10, n_inner_reps=5):
    """
    Full double-CV protocol from Wessels et al. 2005.

    Returns
    -------
    results : dict with keys 'train_perf', 'val_perf' (arrays over 300 folds),
              'best_k_per_fold', 'k_inner_train', 'k_inner_val',
              'k_inner_train_pc', 'k_inner_val_pc' (per-class inner curves)
    """
    classes = np.unique(y)
    train_perfs, val_perfs, best_ks, fold_features = [], [], [], []
    # Accumulate inner curves across all outer reps
    agg_inner_train = {k: [] for k in k_candidates}
    agg_inner_val   = {k: [] for k in k_candidates}
    # Per-class inner curves
    agg_inner_train_pc = {cls: {k: [] for k in k_candidates} for cls in classes}
    agg_inner_val_pc   = {cls: {k: [] for k in k_candidates} for cls in classes}

    for rep in tqdm(range(n_outer_reps), desc=f"{model_name} outer reps"):
        skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=rep)
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # ── Inner loop: find best k ──────────────────────────────
            best_k, inn_train, inn_val, inn_train_pc, inn_val_pc = inner_loop(
                X_tr, y_tr, model_name, k_candidates,
                n_inner_folds=n_inner_folds, n_inner_reps=n_inner_reps
            )
            for k in k_candidates:
                agg_inner_train[k].append(inn_train[k])
                agg_inner_val[k].append(inn_val[k])
            for cls in classes:
                if cls in inn_train_pc:
                    for k in k_candidates:
                        agg_inner_train_pc[cls][k].append(inn_train_pc[cls][k])
                        agg_inner_val_pc[cls][k].append(inn_val_pc[cls][k])

            best_ks.append(best_k)

            # ── Train final predictor on full training set ────────────
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            k_safe   = min(best_k, X_tr_sc.shape[1])
            sel_idx  = relief_f_top_k(X_tr_sc, y_tr, k_safe)
            fold_features.append(sel_idx.copy())
            X_tr_k   = X_tr_sc[:, sel_idx]
            X_te_k   = X_te_sc[:, sel_idx]

            model = get_model(model_name)
            model.fit(X_tr_k, y_tr)

            train_perfs.append(avg_sens_spec(y_tr, model.predict(X_tr_k)))
            val_perfs.append(avg_sens_spec(y_te, model.predict(X_te_k)))

        if (rep + 1) % 10 == 0:
            print(f"  [{model_name}] rep {rep+1}/{n_outer_reps}  "
                  f"val={np.mean(val_perfs):.3f} ± {np.std(val_perfs):.3f}")

    return {
        "train_perf":    np.array(train_perfs),
        "val_perf":      np.array(val_perfs),
        "best_k_per_fold": np.array(best_ks),
        "k_inner_train": {k: np.mean(agg_inner_train[k]) for k in k_candidates},
        "k_inner_val":   {k: np.mean(agg_inner_val[k])   for k in k_candidates},
        "k_inner_train_pc": {cls: {k: np.mean(agg_inner_train_pc[cls][k]) if agg_inner_train_pc[cls][k] else 0.0
                                   for k in k_candidates} for cls in classes},
        "k_inner_val_pc":   {cls: {k: np.mean(agg_inner_val_pc[cls][k]) if agg_inner_val_pc[cls][k] else 0.0
                                   for k in k_candidates} for cls in classes},
        # Raw per-fold lists for CI computation
        "k_inner_train_raw": {k: list(agg_inner_train[k]) for k in k_candidates},
        "k_inner_val_raw":   {k: list(agg_inner_val[k])   for k in k_candidates},
        "k_inner_train_pc_raw": {cls: {k: list(agg_inner_train_pc[cls][k])
                                        for k in k_candidates} for cls in classes},
        "k_inner_val_pc_raw":   {cls: {k: list(agg_inner_val_pc[cls][k])
                                        for k in k_candidates} for cls in classes},
        "fold_features": fold_features,
    }


# ─────────────────────────────────────────────
# 4.  Run 
# ─────────────────────────────────────────────

def run_all(n_outer_reps=100):
    df_features = pd.read_csv('Train_call.txt', sep=None, engine='python')
    df_labels = pd.read_csv('Train_clinical.txt', sep=None, engine='python')
    df_features = df_features.drop_duplicates()

    X_joined = df_features.T.reset_index().rename(columns={'index': 'Sample'}).merge(df_labels, how='inner', on='Sample')

    # Split  training and testing 
    feature_df = X_joined.drop(columns=['Subgroup', 'Sample'])
    feature_names = feature_df.columns.to_numpy()
    X = feature_df.to_numpy(dtype=float)
    y = X_joined['Subgroup'].to_numpy()

    # Keep genomic coordinates for saved features (Chromosome, Start, End)
    feature_info = df_features[['Chromosome', 'Start', 'End']].reset_index(drop=True)




    max_features = X.shape[1] 
    k_candidates = np.unique(np.geomspace(1, max_features, num=15).astype(int)).tolist()

    print(k_candidates)
    
    model_names  = ["KNN", "SVM", "RF"]
    all_results  = {}

    for mname in model_names:
        print(f"\n{'='*55}")
        print(f" Running  pipeline for {mname}-model")
        print(f"{'='*55}")
        all_results[mname] = wessels_pipeline(
            X, y, mname, k_candidates,
            n_outer_folds=3, n_outer_reps=n_outer_reps,
            n_inner_folds=10, n_inner_reps=5
        )

    save_best_val_features(all_results, X, y, feature_info, output_dir="output")

    return all_results, k_candidates


# ─────────────────────────────────────────────
# 5.  Plots
# ─────────────────────────────────────────────

def _ci95(values):
    """Return (mean, lower, upper) for 95% confidence interval."""
    arr = np.asarray(values)
    n = len(arr)
    mean = np.mean(arr)
    if n < 2:
        return mean, mean, mean
    se = np.std(arr, ddof=1) / np.sqrt(n)
    hw = 1.96 * se
    return mean, mean - hw, mean + hw


def plot_results(all_results, k_candidates, output_dir="output"):
    models = list(all_results.keys())
    model_colors = {"RF": "#1D0572", "KNN": "#1DAB51", "SVM": "#CA8919"}
    class_colors_map = {"HER2+": "#E63946", "HR+": "#457B9D", "Triple Neg": "#2A9D8F"}
    fallback_colors = plt.cm.Set1.colors
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    x_pos = np.arange(len(k_candidates))

    def _get_class_color(cls, ci):
        return class_colors_map.get(cls, fallback_colors[ci % len(fallback_colors)])

    def _style_ax(ax, fig):
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F1F5F9")
        ax.grid(True, color="white", lw=1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    for mname in models:
        res = all_results[mname]
        mc = model_colors.get(mname, "#333333")
        best_k_val = max(res["k_inner_val"], key=res["k_inner_val"].get)
        classes = sorted(res["k_inner_val_pc"].keys())

        # ── One plot per class per model ──────────────────────────
        for ci, cls in enumerate(classes):
            cc = _get_class_color(cls, ci)
            best_k_cls = max(res["k_inner_val_pc"][cls], key=res["k_inner_val_pc"][cls].get)

            tr_mean, tr_lo, tr_hi = [], [], []
            va_mean, va_lo, va_hi = [], [], []
            for k in k_candidates:
                m, lo, hi = _ci95(res["k_inner_train_pc_raw"][cls][k])
                tr_mean.append(m); tr_lo.append(lo); tr_hi.append(hi)
                m, lo, hi = _ci95(res["k_inner_val_pc_raw"][cls][k])
                va_mean.append(m); va_lo.append(lo); va_hi.append(hi)

            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
            _style_ax(ax, fig)

            ax.fill_between(x_pos, tr_lo, tr_hi, color=cc, alpha=0.12)
            ax.plot(x_pos, tr_mean, "s--", color=cc, lw=2, ms=4, label="Train")
            ax.fill_between(x_pos, va_lo, va_hi, color=cc, alpha=0.22)
            ax.plot(x_pos, va_mean, "o-", color=cc, lw=2, ms=4, label="Validation")

            if best_k_cls in {k: i for i, k in enumerate(k_candidates)}:
                pos = {k: i for i, k in enumerate(k_candidates)}[best_k_cls]
                ax.axvline(pos, color=cc, lw=1.4, ls="--", alpha=0.7)
                ax.text(pos + 0.2, ax.get_ylim()[0] or 0.5,
                        f"k*={best_k_cls}", fontsize=8, color=cc, va="bottom", rotation=90)

            ax.set_title(f"{mname} · {cls}\nk* = {best_k_cls}   (95% CI shaded)",
                         fontsize=11, fontweight="bold", color="#1E293B", pad=6)
            ax.set_xlabel("k  (number of Relief-F features)", fontsize=9)
            ax.set_ylabel("(Sensitivity + Specificity) / 2", fontsize=9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(k_candidates)
            ax.legend(fontsize=9, framealpha=0.6)

            cls_safe = str(cls).replace(" ", "_").replace("/", "-")
            p = os.path.join(output_dir, f"{mname.lower()}_k_vs_perf_{cls_safe}.png")
            fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            saved_paths.append(p)
            print(f"Figure saved -> {p}")

        # ── Average (all-class) plot per model ────────────────────
        tr_mean, tr_lo, tr_hi = [], [], []
        va_mean, va_lo, va_hi = [], [], []
        for k in k_candidates:
            m, lo, hi = _ci95(res["k_inner_train_raw"][k])
            tr_mean.append(m); tr_lo.append(lo); tr_hi.append(hi)
            m, lo, hi = _ci95(res["k_inner_val_raw"][k])
            va_mean.append(m); va_lo.append(lo); va_hi.append(hi)

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
        _style_ax(ax, fig)

        ax.fill_between(x_pos, tr_lo, tr_hi, color=mc, alpha=0.12)
        ax.plot(x_pos, tr_mean, "s--", color=mc, lw=2.2, ms=4, label="Train")
        ax.fill_between(x_pos, va_lo, va_hi, color=mc, alpha=0.22)
        ax.plot(x_pos, va_mean, "o-", color=mc, lw=2.2, ms=4, label="Validation")

        if best_k_val in {k: i for i, k in enumerate(k_candidates)}:
            pos = {k: i for i, k in enumerate(k_candidates)}[best_k_val]
            ax.axvline(pos, color=mc, lw=1.6, ls="--", alpha=0.8)
            ax.text(pos + 0.2, ax.get_ylim()[0] or 0.5,
                    f"k*={best_k_val}", fontsize=8, color=mc, va="bottom", rotation=90)

        ax.set_title(f"{mname} · Average (all classes)\nk* = {best_k_val}   (95% CI shaded)",
                     fontsize=11, fontweight="bold", color="#1E293B", pad=6)
        ax.set_xlabel("k  (number of Relief-F features)", fontsize=9)
        ax.set_ylabel("(Sensitivity + Specificity) / 2", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(k_candidates)
        ax.legend(fontsize=9, framealpha=0.6)

        p = os.path.join(output_dir, f"{mname.lower()}_k_vs_perf_average.png")
        fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        saved_paths.append(p)
        print(f"Figure saved -> {p}")

    return saved_paths


# ─────────────────────────────────────────────
# 6.  Summary 
# ─────────────────────────────────────────────

def print_summary(all_results, k_candidates):
    print("\n" + "="*65)
    print(f"{'Model':<6} {'Best k(val)':>11} {'Best k*':>8} {'Train mean':>12} {'Train SD':>10} "
                    f"{'Val mean':>10} {'Val SD':>8}")
    print("-"*65)
    for mname, res in all_results.items():
            best_k_val = max(res["k_inner_val"], key=res["k_inner_val"].get)
            best_k = int(np.round(np.mean(res["best_k_per_fold"])))
            print(f"{mname:<6} {best_k_val:>11} {best_k:>8} "
              f"{np.mean(res['train_perf']):>12.4f} "
              f"{np.std(res['train_perf']):>10.4f} "
              f"{np.mean(res['val_perf']):>10.4f} "
              f"{np.std(res['val_perf']):>8.4f}")
            # Per-class best k
            if "k_inner_val_pc" in res:
                for cls in sorted(res["k_inner_val_pc"].keys()):
                    best_k_cls = max(res["k_inner_val_pc"][cls],
                                     key=res["k_inner_val_pc"][cls].get)
                    best_val_cls = res["k_inner_val_pc"][cls][best_k_cls]
                    print(f"  └─ class '{cls}': best k(val)={best_k_cls}  "
                          f"val score={best_val_cls:.4f}")
    print("="*65)


# ─────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Set n_outer_reps=100 for the full protocol; use 10 for a quick test
    N_REPS = 100 
    results, k_cands = run_all(n_outer_reps=N_REPS)
    print_summary(results, k_cands)
    plot_results(results, k_cands)