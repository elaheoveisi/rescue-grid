

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "igaze" / "igaze"))


CONFIG = ROOT / "configs" / "config_analysis.yml"


# ---------------------------------------------------------------------------
# AOI helpers
# ---------------------------------------------------------------------------

def assign_aoi(x_px: float, y_px: float, aois: list[dict]) -> str:
    """Return the name of the first AOI that contains (x_px, y_px), else 'offscreen'."""
    for aoi in aois:
        if aoi["x_min"] <= x_px <= aoi["x_max"] and aoi["y_min"] <= y_px <= aoi["y_max"]:
            return aoi["name"]
    return "offscreen"


def aoi_transition_matrix(fix_aoi_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:

    labels   = [a["name"] for a in aois]
    matrix   = pd.DataFrame(0, index=labels, columns=labels)
    sequence = fix_aoi_df[fix_aoi_df["aoi"] != "offscreen"]["aoi"].tolist()
    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix


def label_fixations(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Add an 'aoi' column to a fixations DataFrame (columns: x, y in pixels)."""
    fix_df = fix_df.copy()
    fix_df["aoi"] = fix_df.apply(lambda r: assign_aoi(r["x"], r["y"], aois), axis=1)
    return fix_df


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_trials(subject_id: str,
                intermediate_dir: Path,
                processed_dir: Path) -> dict[str, dict]:

    sub_proc = processed_dir / f"sub-{subject_id}"
    sub_int  = intermediate_dir / f"sub-{subject_id}"

    if not sub_int.exists():
        return {}

    data = {}
    for trial_dir in sorted(sub_int.iterdir()):
        if not trial_dir.is_dir():
            continue

        # Process all runs, not just _best

        eye_h5 = trial_dir / "eyetracker.h5"
        if not eye_h5.exists():
            continue

        fix_h5 = sub_proc / trial_dir.name / "fixations.h5"
        fix_df = pd.read_hdf(fix_h5, key="fixations") if fix_h5.exists() else pd.DataFrame()

        data[trial_dir.name] = {
            "fixations":  fix_df,
            "eyetracker": pd.read_hdf(eye_h5, key="eyetracker"),
        }
    return data


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(subject_id: str,
                    intermediate_dir: Path,
                    processed_dir: Path,
                    aois: list[dict]) -> list[dict]:
    data = load_trials(subject_id, intermediate_dir, processed_dir)
    if not data:
        print(f"  [SKIP] {subject_id}: no trial data")
        return []

    print(f"  [SUB]  {subject_id}")
    summaries = []

    # --- AOI labelling (per trial) ---
    for trial_id, streams in data.items():
        fix_df = streams["fixations"]

        if fix_df.empty:
            print(f"         {trial_id:35s}  no fixations file — skipping AOI labelling")
            summaries.append({
                "subject": subject_id,
                "trial":   trial_id,
                "n_fixations": 0,
                "n_fixations_offscreen": 0,
                **{f"n_fixations_{a['name']}": 0 for a in aois},
            })
            continue

        fix_aoi = label_fixations(fix_df, aois)

        out_dir = processed_dir / f"sub-{subject_id}" / trial_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fix_aoi.to_csv(out_dir / "fixations_aoi.csv", index=False)
        fix_aoi.to_hdf(out_dir / "fixations_aoi.h5", key="fixations_aoi", mode="w")

        # transition matrix
        trans = aoi_transition_matrix(fix_aoi, aois)
        trans.to_csv(out_dir / "aoi_transitions.csv")

        counts    = fix_aoi["aoi"].value_counts().to_dict()
        dur_by_aoi = fix_aoi.groupby("aoi")["duration_ms"].agg(["sum", "mean"])
        total_dur  = fix_aoi["duration_ms"].sum()
        n_total    = len(fix_aoi)
        n_off      = counts.get("offscreen", 0)

        aoi_stats = {}
        for a in aois:
            n   = counts.get(a["name"], 0)
            dur = float(dur_by_aoi.loc[a["name"], "sum"])  if a["name"] in dur_by_aoi.index else 0.0
            avg = float(dur_by_aoi.loc[a["name"], "mean"]) if a["name"] in dur_by_aoi.index else 0.0
            pct = round(dur / total_dur * 100, 2) if total_dur > 0 else 0.0
            aoi_stats[f"n_fixations_{a['name']}"]  = n
            aoi_stats[f"{a['name']}_total_dur_ms"] = round(dur, 2)
            aoi_stats[f"{a['name']}_mean_dur_ms"]  = round(avg, 2)
            aoi_stats[f"{a['name']}_pct_dur"]      = pct

        aoi_parts = []
        for a in aois:
            name = a["name"]
            aoi_parts.append(f"{name}={aoi_stats['n_fixations_' + name]}({aoi_stats[name + '_pct_dur']}%)")
        print(f"         {trial_id:35s}  total={n_total:>4}  offscreen={n_off:>3}  "
              + "  ".join(aoi_parts))

        summaries.append({
            "subject":     subject_id,
            "trial":       trial_id,
            "n_fixations": n_total,
            "n_fixations_offscreen": n_off,
            **aoi_stats,
        })

    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    subjects     = [str(s) for s in cfg.get("sub", [])]
    intermediate = ROOT / cfg["paths"]["intermediate"]
    processed    = ROOT / cfg["paths"]["processed"]
    eye_cfg      = cfg.get("eyetracker", {})
    aois         = cfg.get("aoi", [])

    if not aois:
        print("WARNING: no AOIs defined in config — all fixations will be labelled 'offscreen'")

    print(f"Processing {len(subjects)} subject(s) | {len(aois)} AOI(s)\n")

    all_summaries = []
    for sid in subjects:
        all_summaries.extend(
            process_subject(sid, intermediate, processed, aois)
        )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        out_path   = processed / "aoi_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nAOI summary -> {out_path.relative_to(ROOT)}")
        print(summary_df.to_string(index=False))

    # aggregate transition matrices across all subjects — overall and per trial type
    aoi_names   = [a["name"] for a in aois]
    total_trans = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
    by_trial    = {}   # base trial name -> aggregated matrix

    for sid in subjects:
        sub_proc = processed / f"sub-{sid}"
        if not sub_proc.exists():
            continue
        for trial_dir in sorted(sub_proc.iterdir()):
            t_file = trial_dir / "aoi_transitions.csv"
            if not t_file.exists():
                continue
            mat = pd.read_csv(t_file, index_col=0)
            mat = mat.reindex(index=aoi_names, columns=aoi_names, fill_value=0)
            total_trans += mat

            # derive base trial name (strip _runN suffix)
            import re as _re
            base = _re.sub(r'_run\d+$', '', trial_dir.name)
            if base not in by_trial:
                by_trial[base] = pd.DataFrame(0, index=aoi_names, columns=aoi_names)
            by_trial[base] += mat

    out_trans = processed / "aoi_transitions_all.csv"
    total_trans.to_csv(out_trans)
    print(f"\nAggregated transitions (all) -> {out_trans.relative_to(ROOT)}")
    print(total_trans.to_string())

    for base, mat in sorted(by_trial.items()):
        out = processed / f"aoi_transitions_{base}.csv"
        mat.to_csv(out)
        print(f"Aggregated transitions ({base}) -> {out.relative_to(ROOT)}")
        print(mat.to_string())

    print("\nDone.")
