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
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────
# 2.  Inner loop — optimize k 
# ─────────────────────────────────────────────

def inner_loop(X_train, y_train, model_name, k_candidates, n_inner_folds=10, n_inner_reps=5):
    """
    10-fold CV (repeated 5×) on the training set to find the best k.
    Returns: best_k, dict{k: mean_train_perf, k: mean_val_perf}
    """
    k_train = {k: [] for k in k_candidates}
    k_val   = {k: [] for k in k_candidates}

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

                k_train[k].append(avg_sens_spec(y_tr, model.predict(X_tr_k)))
                k_val[k].append(avg_sens_spec(y_te, model.predict(X_te_k)))

    mean_val = {k: np.mean(k_val[k]) for k in k_candidates}
    best_k   = max(mean_val, key=mean_val.get)
    return best_k, {k: np.mean(k_train[k]) for k in k_candidates}, mean_val


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
              'best_k_per_fold', 'k_inner_train', 'k_inner_val'
    """
    train_perfs, val_perfs, best_ks = [], [], []
    # Accumulate inner curves across all outer reps
    agg_inner_train = {k: [] for k in k_candidates}
    agg_inner_val   = {k: [] for k in k_candidates}

    for rep in range(n_outer_reps):
        skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=rep)
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # ── Inner loop: find best k ──────────────────────────────
            best_k, inn_train, inn_val = inner_loop(
                X_tr, y_tr, model_name, k_candidates,
                n_inner_folds=n_inner_folds, n_inner_reps=n_inner_reps
            )
            for k in k_candidates:
                agg_inner_train[k].append(inn_train[k])
                agg_inner_val[k].append(inn_val[k])

            best_ks.append(best_k)

            # ── Train final predictor on full training set ────────────
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            k_safe   = min(best_k, X_tr_sc.shape[1])
            sel_idx  = relief_f_top_k(X_tr_sc, y_tr, k_safe)
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
    }


# ─────────────────────────────────────────────
# 4.  Run 
# ─────────────────────────────────────────────

def run_all(n_outer_reps=100):
    # Use breast-cancer dataset as a stand-in (replace with your own data)
    df_features = pd.read_csv('Train_call.txt', sep=None, engine='python')
    df_labels = pd.read_csv('Train_clinical.txt', sep=None, engine='python')
    df_features = df_features.drop_duplicates()

    X_joined = df_features.T.reset_index().rename(columns={'index': 'Sample'}).merge(df_labels, how='inner', on='Sample')

    # Split  training and testing 
    X = X_joined.drop(columns=['Subgroup', 'Sample']).to_numpy(dtype=float)
    y = X_joined['Subgroup'].to_numpy()




    max_features = X.shape[1] 
    k_candidates = np.unique(np.geomspace(1, max_features, num=15).astype(int)).tolist()

    print(k_candidates)
    
    model_names  = ["RF"] #, "KNN", "SVM"]
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

    return all_results, k_candidates


# ─────────────────────────────────────────────
# 5.  Plots
# ─────────────────────────────────────────────

def plot_results(all_results, k_candidates, output_dir="output"):
    models     = list(all_results.keys())
    colors     = {"RF": "#1D0572", "KNN": "#1DAB51", "SVM": "#CA8919"}
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for mname in models:
        res = all_results[mname]
        c = colors[mname]
        best_k_val = max(res["k_inner_val"], key=res["k_inner_val"].get)
        best_k = int(np.round(np.mean(res["best_k_per_fold"])))
        k_arr = np.array(k_candidates)
        x_pos = np.arange(len(k_candidates))
        k_to_pos = {k: i for i, k in enumerate(k_candidates)}
        tr_arr = np.array([res["k_inner_train"][k] for k in k_candidates])
        va_arr = np.array([res["k_inner_val"][k] for k in k_candidates])
        tr = res["train_perf"]
        va = res["val_perf"]
        

        # Plot 1: k vs inner-CV performance
        fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
        fig1.patch.set_facecolor("#F8FAFC")
        ax1.plot(x_pos, tr_arr, "o-", color=c, lw=2, ms=3.5, label="Train (inner CV)")
        ax1.plot(x_pos, va_arr, "s--", color=c, lw=2, ms=3.5, alpha=0.65, label="Val (inner CV)")
        if best_k_val in k_to_pos:
            ax1.axvline(k_to_pos[best_k_val], color="#CA8919", lw=1.6, ls="--", alpha=0.95, label=f"Best k(val)={best_k_val}")
        #if best_k in k_to_pos:
         #   ax1.axvline(k_to_pos[best_k], color=c, lw=1.4, ls=":", alpha=0.9, label=f"Best k(train)={best_k}")
        #ax1.text(best_k + 0.3, ax1.get_ylim()[0] + 0.005, f"k(train)={best_k}", fontsize=8, color=c, va="bottom")
        #ax1.text(best_k_val + 0.3, ax1.get_ylim()[0] + 0.005, f"k(val)={best_k_val}", fontsize=8, color="#CA8919", va="bottom")
        ax1.set_title(
            f"{mname} · k vs Performance\nBest k(val)={best_k_val} ",
            fontsize=11,
            fontweight="bold",
            color="#1E293B",
            pad=6,
        )
        ax1.set_xlabel("k  (number of Relief-F features)", fontsize=9)
        ax1.set_ylabel("Avg(Sensitivity, Specificity)", fontsize=9)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(k_candidates)
        ax1.legend(fontsize=8, framealpha=0.6)
        ax1.set_facecolor("#F1F5F9")
        ax1.grid(True, color="white", lw=1.2)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.tick_params(labelsize=8)

        k_plot_path = os.path.join(output_dir, f"{mname.lower()}_k_vs_performance.png")
        fig1.savefig(k_plot_path, bbox_inches="tight", facecolor=fig1.get_facecolor())
        plt.close(fig1)
        saved_paths.append(k_plot_path)
        print(f"Figure saved -> {k_plot_path}")

        # Plot 2: outer-CV train vs validation distribution
        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
        fig2.patch.set_facecolor("#F8FAFC")
        bp = ax2.boxplot([tr, va], patch_artist=True, widths=0.4,
                         medianprops=dict(color="white", lw=2.5),
                         whiskerprops=dict(color="#64748B"),
                         capprops=dict(color="#64748B"),
                         flierprops=dict(marker=".", color=c, alpha=0.4, ms=4))
        for patch, alpha in zip(bp["boxes"], [0.85, 0.45]):
            patch.set_facecolor(c)
            patch.set_alpha(alpha)

        for i, arr in enumerate([tr, va], start=1):
            ax2.scatter(i, np.mean(arr), zorder=5, color="white", edgecolors=c, s=60, linewidths=1.5)

        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["Train", "Validation"], fontsize=9)
        ax2.set_ylabel("Avg(Sensitivity, Specificity)", fontsize=9)
        ax2.set_title(
            f"{mname} · Outer-CV Distribution\n"
            f"Best k(val)={best_k_val}   Best k(train)={best_k}\n"
            f"Train {np.mean(tr):.3f}±{np.std(tr):.3f}   "
            f"Val {np.mean(va):.3f}±{np.std(va):.3f}",
            fontsize=10, fontweight="bold", color="#1E293B", pad=6
        )
        ax2.set_facecolor("#F1F5F9")
        ax2.grid(True, axis="y", color="white", lw=1.2)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(labelsize=8)
        ax2.set_ylim(0.5, 1.05)

        dist_plot_path = os.path.join(output_dir, f"{mname.lower()}_outer_cv_distribution.png")
        fig2.savefig(dist_plot_path, bbox_inches="tight", facecolor=fig2.get_facecolor())
        plt.close(fig2)
        saved_paths.append(dist_plot_path)
        print(f"Figure saved -> {dist_plot_path}")

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
    print("="*65)


# ─────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Set n_outer_reps=100 for the full protocol; use 10 for a quick test
    N_REPS = 10
    results, k_cands = run_all(n_outer_reps=N_REPS)
    print_summary(results, k_cands)
    plot_results(results, k_cands)