"""
Bayesian hyperparameter optimisation (Optuna) for RF, KNN, and SVM.

Uses Relief-F feature selection (top-k) inside each CV fold.
Logs accuracy, per-class sensitivity, and overall sensitivity
for every trial in the optimisation trajectory.

Optuna's TPE sampler + median pruning makes this much faster than
exhaustive grid search while exploring the same (or wider) space.
"""

import numpy as np
import pandas as pd
import os
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from ReliefF import ReliefF
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)



def avg_sens_spec(y_true, y_pred):
    """Average one-vs-rest (sensitivity + specificity) / 2 across classes."""
    class_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    per_class_scores = []
    for label in class_labels:
        y_true_bin = np.asarray(y_true) == label
        y_pred_bin = np.asarray(y_pred) == label
        tn, fp, fn, tp = confusion_matrix(
            y_true_bin, y_pred_bin, labels=[False, True]
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


# ── Relief-F wrapper ─────────────────────────────────────────

def relief_f_top_k(X, y, k):
    n_neighbors = min(10, len(y) - 1)
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=k)
    fs.fit(X, y)
    return fs.top_features[:k]


# ── Optuna search spaces ────────────────────────────────────

def suggest_rf(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 500]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_categorical("max_depth", ["None", "5", "10", "20"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

def _parse_rf(params):
    p = dict(params)
    p["max_depth"] = None if p["max_depth"] == "None" else int(p["max_depth"])
    return p


def suggest_knn(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 15, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    }


def suggest_svm(trial):
    return {
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


SUGGEST_FN = {"RF": suggest_rf, "KNN": suggest_knn, "SVM": suggest_svm}


def make_model(name, params):
    if name == "RF":
        return RandomForestClassifier(random_state=42, **_parse_rf(params))
    elif name == "KNN":
        return KNeighborsClassifier(**params)
    elif name == "SVM":
        return SVC(**params)
    raise ValueError(name)


# ── Objective with pruning ───────────────────────────────────

def make_objective(X, y, model_name, k_features, n_splits, n_reps, classes):
    """Return an Optuna objective that supports pruning across CV reps."""

    def objective(trial):
        params = SUGGEST_FN[model_name](trial)
        accs, scores = [], []
        pc_scores = {cls: [] for cls in classes}

        for rep in range(n_reps):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
            for tr_idx, te_idx in skf.split(X, y):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

                k_safe = min(k_features, X_tr.shape[1])
                sel = relief_f_top_k(X_tr, y_tr, k_safe)

                model = make_model(model_name, params)
                model.fit(X_tr[:, sel], y_tr)
                y_pred = model.predict(X_te[:, sel])

                accs.append(accuracy_score(y_te, y_pred))
                scores.append(avg_sens_spec(y_te, y_pred))
                pc = per_class_sens_spec(y_te, y_pred)
                for cls in classes:
                    pc_scores[cls].append(pc.get(cls, 0.0))

            # Report intermediate value after each rep → enables pruning
            trial.report(np.mean(scores), rep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Store extra metrics as user attrs so we can retrieve them later
        trial.set_user_attr("accuracy_mean", float(np.mean(accs)))
        trial.set_user_attr("accuracy_std", float(np.std(accs)))
        trial.set_user_attr("avg_sens_spec_mean", float(np.mean(scores)))
        trial.set_user_attr("avg_sens_spec_std", float(np.std(scores)))
        for cls in classes:
            trial.set_user_attr(f"sens_spec_{cls}_mean", float(np.mean(pc_scores[cls])))
            trial.set_user_attr(f"sens_spec_{cls}_std", float(np.std(pc_scores[cls])))

        return float(np.mean(scores))  # maximize this

    return objective


# ── Optuna study driver ──────────────────────────────────────

def optuna_search(X, y, model_name, k_features,
                  n_splits=10, n_reps=5, n_trials=50,
                  output_dir="output"):
    classes = np.unique(y)

    print(f"\n{'='*55}")
    print(f" Optuna search for {model_name}  ({n_trials} trials)")
    print(f"{'='*55}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    objective = make_objective(X, y, model_name, k_features, n_splits, n_reps, classes)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Build results DataFrame from all completed trials ────
    records = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rec = {"trial": t.number, "value": t.value}
        for k, v in t.params.items():
            rec[f"param_{k}"] = v
        for k, v in t.user_attrs.items():
            rec[k] = v
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name.lower()}_optuna_search.csv")
    df.to_csv(path, index=False)

    best = study.best_trial
    print(f"Results saved -> {path}")
    print(f"Best trial #{best.number}:  {best.params}")
    print(f"  accuracy        = {best.user_attrs['accuracy_mean']:.4f} "
          f"± {best.user_attrs['accuracy_std']:.4f}")
    print(f"  avg(sens,spec)  = {best.user_attrs['avg_sens_spec_mean']:.4f} "
          f"± {best.user_attrs['avg_sens_spec_std']:.4f}")
    for cls in classes:
        print(f"  (sens+spec)/2 ({cls}) = {best.user_attrs[f'sens_spec_{cls}_mean']:.4f} "
              f"± {best.user_attrs[f'sens_spec_{cls}_std']:.4f}")
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  ({n_pruned}/{n_trials} trials pruned)")

    return df, study


# ── Optimization trajectory plots ────────────────────────────

def plot_optimization(study, model_name, output_dir="output"):
    """Plot per-trial score + running best, accuracy trajectory, and per-class breakdown."""
    os.makedirs(output_dir, exist_ok=True)
    model_colors = {"RF": "#1D0572", "KNN": "#1DAB51", "SVM": "#CA8919"}
    mc = model_colors.get(model_name, "#333333")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"  No completed trials for {model_name}, skipping plots.")
        return []

    trials_sorted = sorted(completed, key=lambda t: t.number)
    trial_nums = [t.number for t in trials_sorted]
    values = [t.value for t in trials_sorted]
    running_best = np.maximum.accumulate(values)
    accs = [t.user_attrs["accuracy_mean"] for t in trials_sorted]

    # Discover class names from first trial's user attrs
    cls_keys = sorted(
        {k.replace("sens_spec_", "").replace("_mean", "")
         for k in trials_sorted[0].user_attrs if k.startswith("sens_spec_") and k.endswith("_mean")}
    )
    class_colors = plt.cm.Set1.colors

    saved = []

    # ── Plot 1: optimization trajectory (score + running best) ──
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F1F5F9")

    ax.scatter(trial_nums, values, c=mc, alpha=0.45, s=30, label="Trial score", zorder=3)
    ax.plot(trial_nums, running_best, color=mc, lw=2.2, label="Best so far", zorder=4)
    ax.set_xlabel("Trial", fontsize=10)
    ax.set_ylabel("Avg (Sens + Spec) / 2", fontsize=10)
    ax.set_title(f"{model_name} · Optimization Trajectory\n"
                 f"Best = {running_best[-1]:.4f}  (trial #{trials_sorted[np.argmax(values)].number})",
                 fontsize=11, fontweight="bold", color="#1E293B")
    ax.legend(fontsize=9, framealpha=0.6)
    ax.grid(True, color="white", lw=1.2)
    ax.spines[["top", "right"]].set_visible(False)

    p1 = os.path.join(output_dir, f"{model_name.lower()}_optuna_trajectory.png")
    fig.savefig(p1, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    saved.append(p1)

    # ── Plot 2: accuracy + overall score side by side ───────────
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    fig2.patch.set_facecolor("#F8FAFC")
    ax2.set_facecolor("#F1F5F9")

    ax2.scatter(trial_nums, accs, alpha=0.4, s=25, color="#6366F1", label="Accuracy", zorder=3)
    ax2.scatter(trial_nums, values, alpha=0.4, s=25, color=mc, label="Avg(Sens,Spec)/2", zorder=3)
    ax2.plot(trial_nums, np.maximum.accumulate(accs), color="#6366F1", lw=1.8, ls="--", label="Best accuracy")
    ax2.plot(trial_nums, running_best, color=mc, lw=1.8, label="Best avg(sens,spec)/2")
    ax2.set_xlabel("Trial", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title(f"{model_name} · Accuracy vs Avg(Sens,Spec)/2",
                  fontsize=11, fontweight="bold", color="#1E293B")
    ax2.legend(fontsize=8, framealpha=0.6, ncol=2)
    ax2.grid(True, color="white", lw=1.2)
    ax2.spines[["top", "right"]].set_visible(False)

    p2 = os.path.join(output_dir, f"{model_name.lower()}_optuna_acc_vs_score.png")
    fig2.savefig(p2, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    saved.append(p2)

    # ── Plot 3: per-class (sens+spec)/2 trajectory ──────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5), dpi=150)
    fig3.patch.set_facecolor("#F8FAFC")
    ax3.set_facecolor("#F1F5F9")

    for ci, cls in enumerate(cls_keys):
        cc = class_colors[ci % len(class_colors)]
        cls_vals = [t.user_attrs[f"sens_spec_{cls}_mean"] for t in trials_sorted]
        ax3.scatter(trial_nums, cls_vals, alpha=0.35, s=20, color=cc, zorder=3)
        ax3.plot(trial_nums, np.maximum.accumulate(cls_vals),
                 color=cc, lw=1.8, label=f"{cls} best")

    ax3.set_xlabel("Trial", fontsize=10)
    ax3.set_ylabel("(Sens + Spec) / 2", fontsize=10)
    ax3.set_title(f"{model_name} · Per-Class Optimization Trajectory",
                  fontsize=11, fontweight="bold", color="#1E293B")
    ax3.legend(fontsize=8, framealpha=0.6)
    ax3.grid(True, color="white", lw=1.2)
    ax3.spines[["top", "right"]].set_visible(False)

    p3 = os.path.join(output_dir, f"{model_name.lower()}_optuna_per_class.png")
    fig3.savefig(p3, bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close(fig3)
    saved.append(p3)

    for p in saved:
        print(f"Figure saved -> {p}")
    return saved


# ── Data loading (same as cross_val_pipeline) ────────────────

def load_data():
    df_features = pd.read_csv("Train_call.txt", sep=None, engine="python")
    df_labels = pd.read_csv("Train_clinical.txt", sep=None, engine="python")
    df_features = df_features.drop_duplicates()

    X_joined = (
        df_features.T.reset_index()
        .rename(columns={"index": "Sample"})
        .merge(df_labels, how="inner", on="Sample")
    )

    feature_df = X_joined.drop(columns=["Subgroup", "Sample"])
    X = feature_df.to_numpy(dtype=float)
    y = X_joined["Subgroup"].to_numpy()
    return X, y


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
          f"classes: {np.unique(y)}")

    K_FEATURES = 30
    N_SPLITS = 10
    N_REPS = 2
    N_TRIALS = 50   # Optuna typically converges in 30-50 trials

    results = {}
    for model_name in ["RF"]: #, "KNN", "SVM"]:
        df, study = optuna_search(
            X, y, model_name, k_features=K_FEATURES,
            n_splits=N_SPLITS, n_reps=N_REPS, n_trials=N_TRIALS,
            output_dir="output",
        )
        results[model_name] = (df, study)
        plot_optimization(study, model_name, output_dir="output")

    # Print summary of best per model
    print("\n" + "=" * 75)
    print(f"{'Model':<6} {'Accuracy':>12} {'Avg(Sens,Spec)':>16} {'Best Params'}")
    print("-" * 75)
    for mname, (df, study) in results.items():
        b = study.best_trial
        params_str = ", ".join(f"{k}={v}" for k, v in b.params.items())
        print(f"{mname:<6} "
              f"{b.user_attrs['accuracy_mean']:>8.4f}±{b.user_attrs['accuracy_std']:.4f}"
              f" {b.user_attrs['avg_sens_spec_mean']:>12.4f}±{b.user_attrs['avg_sens_spec_std']:.4f}"
              f"  {params_str}")
    print("=" * 75)
