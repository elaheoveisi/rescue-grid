import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "config_analysis.yml"

sys.path.insert(0, str(ROOT))
from analysis.features.extract_features import process_subject  # noqa: E402


def load_features(cfg):
    subjects         = [str(s) for s in cfg.get("sub", [])]
    intermediate_dir = ROOT / cfg["paths"]["intermediate"]
    processed_dir    = ROOT / cfg["paths"]["processed"]
    missing_val      = cfg.get("eyetracker", {}).get("missing", 0.0)
    expertise_map    = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}

    checkpoints = cfg.get("checkpoints", [])
    fix_cfg     = cfg.get("fixation", {})
    fix_mindur  = fix_cfg.get("mindur", 50.0)
    fix_maxdur  = fix_cfg.get("maxdur", 400.0)
    sac_cfg     = cfg.get("saccade", {})
    sac_mindur  = sac_cfg.get("mindur", 10.0)
    sac_maxdur  = sac_cfg.get("maxdur", 150.0)

    all_rows = []
    for sid in subjects:
        all_rows.extend(
            process_subject(sid, intermediate_dir, processed_dir, missing_val,
                            expertise=expertise_map.get(sid, "unknown"),
                            checkpoints=checkpoints,
                            fix_mindur=fix_mindur,
                            fix_maxdur=fix_maxdur,
                            sac_mindur=sac_mindur,
                            sac_maxdur=sac_maxdur)
        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # keep only best run per (subject, base_trial)
    df["final_saved_victims"] = pd.to_numeric(
        df.get("final_saved_victims"), errors="coerce"
    ).fillna(0)
    df = (
        df.sort_values(
            ["subject", "base_trial", "final_saved_victims", "run"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["subject", "base_trial"], keep="first")
        .reset_index(drop=True)
    )
    df["trial"] = df["base_trial"]

    return df


def _fixed_effects_formula(outcome, df):
    formula = f"{outcome} ~ C(trial)"
    if "expertise" in df.columns and df["expertise"].nunique() > 1:
        formula += " + C(expertise)"
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


def run_glmm_models(df, continuous_outcomes, count_outcomes):
    mixed = _run_outcome_models(df, continuous_outcomes, _fit_mixedlm)
    count = _run_outcome_models(df, count_outcomes, _fit_poisson_glmm)
    
    mixed_rows = []
    for outcome, model in mixed.items():
        for term in model.params.index:
            mixed_rows.append({
                "outcome": outcome,
                "term": term,
                "coef": model.params[term],
                "se": model.bse.get(term),
                "p_value": model.pvalues.get(term)
            })
            
    count_rows = []
    for outcome, model in count.items():
        fe_mean = np.asarray(model.fe_mean)
        fe_sd = np.asarray(model.fe_sd)
        for term, coef, se in zip(model.model.exog_names, fe_mean, fe_sd):
            p_value = float(2 * norm.sf(abs(coef / se))) if se > 0 else None
            count_rows.append({
                "outcome": outcome,
                "term": term,
                "coef": coef,
                "se": se,
                "p_value": p_value
            })
            
    return pd.DataFrame(mixed_rows), pd.DataFrame(count_rows)


def run_from_config(config_path=CONFIG, verbose=True):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    continuous_outcomes = cfg.get("continuous_outcomes", [])
    count_outcomes = cfg.get("count_outcomes", [])
    
    if verbose:
        print(f"Loading features for {len(cfg.get('sub', []))} subject(s) from {cfg['paths']['intermediate']} ...")
        
    df = load_features(cfg)
    
    if verbose: 
        print(f"  {len(df)} rows  x  {len(df.columns)} columns\n")
        
    processed_dir = ROOT / cfg["paths"]["processed"]
    mixed_df, count_df = run_glmm_models(df, continuous_outcomes, count_outcomes)
    
    if not mixed_df.empty:
        out_path = processed_dir / "glmm_mixed_results.csv"
        mixed_df.to_csv(out_path, index=False)
        if verbose: 
            print(f"Mixed LMM results -> {out_path.relative_to(ROOT)}\n{mixed_df.to_string(index=False)}")
            
    if not count_df.empty:
        out_path = processed_dir / "glmm_count_results.csv"
        count_df.to_csv(out_path, index=False)
        if verbose: 
            print(f"\nCount (Poisson GLMM) results -> {out_path.relative_to(ROOT)}\n{count_df.to_string(index=False)}")
            
    if mixed_df.empty and count_df.empty and verbose:
        print("No models were fitted — check that intermediate data exists for subjects in config.")
        
    all_outcomes = count_outcomes + continuous_outcomes
    available = [c for c in all_outcomes if c in df.columns]
    
    means_df = df.groupby("trial")[available].mean().round(3)
    out_path = processed_dir / "outcome_means_by_trial.csv"
    means_df.to_csv(out_path)
    
    if verbose: 
        print(f"\nOutcome means by trial -> {out_path.relative_to(ROOT)}\n{means_df.to_string()}\nDone.")
        
    return mixed_df, count_df, means_df
