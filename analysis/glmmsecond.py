from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT   = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "config_analysis.yml"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

processed_dir = ROOT / cfg["paths"]["processed"]
glmm2_cfg     = cfg["glmm2"]

# Load dataframes defined in config

dataframes = {}
for dataset in glmm2_cfg["datasets"]:
    for fname in dataset.get("dataframes", []):
        path = processed_dir / fname
        key  = fname.replace(".csv", "")
        dataframes[key] = pd.read_csv(path)


FEATURES = [
    "victims_per_step",
    "n_fixations",
    "mean_fixation_dur_ms",
    "std_pupil_diam",
    "n_saccades",
    "game_area_pct_dur",
    "chat_panel_pct_dur",
    "saved_victims",
    "offscreen_pct_dur",
]

expertise_map = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}


def prepare_df(df):
    """Add condition and expertise columns, keep only FEATURES + id cols."""
    df = df.copy()
    df["condition"] = df["category"].apply(
        lambda c: "no_llm" if c == "dummy" else "llm"
    )
    df["expertise"] = df["participant"].str.replace("sub-", "", regex=False).map(expertise_map).fillna("unknown")
    df["condition"] = pd.Categorical(df["condition"], categories=["no_llm", "llm"])
    df["expertise"] = pd.Categorical(df["expertise"], categories=["novice", "expert"])
    keep = ["participant", "category", "condition", "expertise"] + [
        f for f in FEATURES if f in df.columns
    ]
    return df[keep]


def run_glmm(df, outcome):
    """Fit a mixed LM for one outcome: condition * expertise, random intercept per participant."""
    clean = df.dropna(subset=[outcome]).copy()
    if clean.empty:
        return None
    formula = (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )
    with warnings.catch_warnings(): #run repeated measure and handling warnings
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            model = smf.mixedlm(formula, clean, groups=clean["participant"]).fit()
        except np.linalg.LinAlgError:
            warnings.warn(f"Skipping '{outcome}': singular matrix.", RuntimeWarning)
            return None
    return model


def run_all(verbose=True):
    rows = []
    for name, df in dataframes.items():
        prepared = prepare_df(df)
        if verbose:
            print(f"\n=== {name} ===")
            print(prepared[["condition", "expertise"]].value_counts().to_string())
        for outcome in FEATURES:
            if outcome not in prepared.columns:
                continue
            model = run_glmm(prepared, outcome)
            if model is None:
                continue
            for term in model.params.index:
                rows.append({
                    "dataset":  name,
                    "outcome":  outcome,
                    "term":     term,
                    "coef":     model.params[term],
                    "se":       model.bse.get(term),
                    "p_value":  model.pvalues.get(term),
                })
            if verbose:
                print(f"  {outcome}: fitted")

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr
        out = processed_dir / "glmm2_results.csv"
        results.to_csv(out, index=False)
        if verbose:
            print(f"\nResults saved -> {out.relative_to(ROOT)}")
            print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run_all()
