import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "config_analysis.yml"


def load_features(cfg):
    processed_dir = ROOT / cfg["paths"]["processed"]
    checkpoints   = cfg["checkpoints"]
    expertise_map = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}

    quarters = []
    for pct in checkpoints:
        path = processed_dir / f"features_q{pct}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} — run extract_quarter_features.py first")
        q_df = pd.read_csv(path)
        feature_cols = [c for c in q_df.columns if c not in ("participant", "category", "quarter_pct")]
        q_df = q_df.rename(columns={c: f"q{pct}_{c}" for c in feature_cols})
        quarters.append(q_df)

    # merge all quarters on participant + category
    df = quarters[0]
    for q_df in quarters[1:]:
        df = df.merge(q_df[["participant", "category"] + [c for c in q_df.columns if c not in ("participant", "category", "quarter_pct")]],
                      on=["participant", "category"], how="outer")

    df = df.rename(columns={"participant": "subject", "category": "trial"})
    df["base_trial"] = df["trial"]
    df["condition"]  = df["trial"].apply(lambda t: "no_llm" if t == "dummy" else "llm")
    df["expertise"]  = df["subject"].apply(lambda s: expertise_map.get(s.replace("sub-", ""), "unknown"))

    return df


def _fixed_effects_formula(outcome, df):
    conditions = df["condition"].dropna().unique().tolist()
    ref = "no_llm" if "no_llm" in conditions else conditions[0]
    formula = f"{outcome} ~ C(condition, Treatment(reference='{ref}'))"
    if "expertise" in df.columns and df["expertise"].nunique() > 1:
        formula += " * C(expertise, Treatment(reference='novice'))"
    return formula


def _fit_mixedlm(df, outcome):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        formula = _fixed_effects_formula(outcome, df)
        try:
            return smf.mixedlm(formula, df, groups=df["subject"]).fit()
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Skipping '{outcome}': singular matrix — too few observations "
                f"({len(df)} rows) to estimate the mixed model.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


def _fit_poisson_glmm(df, outcome):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        formula = _fixed_effects_formula(outcome, df)
        ident_dict = {"subject": "0 + C(subject)"}
        return PoissonBayesMixedGLM.from_formula(formula, ident_dict, df).fit_vb()


def _run_outcome_models(df, outcomes, fit_fn):
    results = {}
    for outcome in outcomes:
        if outcome in df.columns and df[outcome].notna().sum() > 0:
            clean_df = df.dropna(subset=[outcome]).copy()
            result = fit_fn(clean_df, outcome)
            if result is not None:
                results[outcome] = result
    return results


def _apply_fdr(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add p_value_fdr column using Benjamini-Hochberg correction."""
    if results_df.empty:
        return results_df
    results_df = results_df.copy()
    valid_mask = results_df["p_value"].notna()
    if valid_mask.sum() == 0:
        results_df["p_value_fdr"] = np.nan
        return results_df
    pvals = results_df.loc[valid_mask, "p_value"].to_numpy()
    _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    results_df["p_value_fdr"] = np.nan
    results_df.loc[valid_mask, "p_value_fdr"] = pvals_fdr
    return results_df


def run_glmm_quarter(df, continuous_outcomes, count_outcomes, quarter):
    """Run mixed LMM + Poisson GLMM for one quarter's outcomes, with FDR."""
    pct_continuous = [f"q{quarter}_{o}" for o in continuous_outcomes]
    pct_count      = [f"q{quarter}_{o}" for o in count_outcomes]

    mixed = _run_outcome_models(df, pct_continuous, _fit_mixedlm)
    count = _run_outcome_models(df, pct_count,      _fit_poisson_glmm)

    mixed_rows = []
    for outcome, model in mixed.items():
        for term in model.params.index:
            mixed_rows.append({
                "quarter":   quarter,
                "outcome":   outcome,
                "term":      term,
                "coef":      model.params[term],
                "se":        model.bse.get(term),
                "p_value":   model.pvalues.get(term),
            })

    count_rows = []
    for outcome, model in count.items():
        fe_mean = np.asarray(model.fe_mean)
        fe_sd   = np.asarray(model.fe_sd)
        for term, coef, se in zip(model.model.exog_names, fe_mean, fe_sd):
            p_value = float(2 * norm.sf(abs(coef / se))) if se > 0 else None
            count_rows.append({
                "quarter":  quarter,
                "outcome":  outcome,
                "term":     term,
                "coef":     coef,
                "se":       se,
                "p_value":  p_value,
            })

    mixed_df = _apply_fdr(pd.DataFrame(mixed_rows))
    count_df = _apply_fdr(pd.DataFrame(count_rows))
    return mixed_df, count_df


def run_from_config(config_path=CONFIG, verbose=True):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    continuous_outcomes = cfg["quarter_continuous_outcomes"]
    count_outcomes      = cfg["quarter_count_outcomes"]
    quarters            = cfg["checkpoints"]

    if verbose:
        print(f"Loading features for {len(cfg.get('sub', []))} subject(s) ...")

    df = load_features(cfg)

    if df.empty:
        print("No features — check that intermediate data exists.")
        return {}, {}

    if verbose:
        print(f"  {len(df)} rows x {len(df.columns)} cols")
        print(f"  condition counts:\n{df['condition'].value_counts().to_string()}\n")

    processed_dir = ROOT / cfg["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_mixed, all_count = [], []

    for q in quarters:
        if verbose:
            print(f"── Quarter {q}% ──────────────────────────────────────")
        mixed_df, count_df = run_glmm_quarter(df, continuous_outcomes, count_outcomes, q)

        if not mixed_df.empty:
            out = processed_dir / f"glmm_q{q}_mixed.csv"
            mixed_df.to_csv(out, index=False)
            if verbose:
                print(f"  Mixed LMM  -> {out.relative_to(ROOT)}")
                print(mixed_df.to_string(index=False))

        if not count_df.empty:
            out = processed_dir / f"glmm_q{q}_count.csv"
            count_df.to_csv(out, index=False)
            if verbose:
                print(f"  Count GLMM -> {out.relative_to(ROOT)}")
                print(count_df.to_string(index=False))

        if mixed_df.empty and count_df.empty and verbose:
            print(f"  [SKIP] no models fitted for quarter {q}")

        all_mixed.append(mixed_df)
        all_count.append(count_df)

    # combined results across all quarters
    combined_mixed = pd.concat(all_mixed, ignore_index=True) if all_mixed else pd.DataFrame()
    combined_count = pd.concat(all_count, ignore_index=True) if all_count else pd.DataFrame()

    if not combined_mixed.empty:
        combined_mixed.to_csv(processed_dir / "glmm_all_mixed.csv", index=False)
    if not combined_count.empty:
        combined_count.to_csv(processed_dir / "glmm_all_count.csv", index=False)

    # means by trial (for all quarter outcomes)
    all_outcome_cols = (
        [f"q{q}_{o}" for q in quarters for o in continuous_outcomes]
        + [f"q{q}_{o}" for q in quarters for o in count_outcomes]
    )
    available = [c for c in all_outcome_cols if c in df.columns]
    means_df = df.groupby("trial")[available].mean().round(3)
    means_df.to_csv(processed_dir / "outcome_means_by_trial.csv")

    if verbose:
        print(f"\nMeans by trial -> {(processed_dir / 'outcome_means_by_trial.csv').relative_to(ROOT)}")
        print(means_df.to_string())
        print("\nDone.")

    return combined_mixed, combined_count


if __name__ == "__main__":
    run_from_config()
