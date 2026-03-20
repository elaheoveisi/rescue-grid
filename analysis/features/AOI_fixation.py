

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
    """
    Load per-trial fixations and raw eyetracker data for a subject.

    Returns:
        data[trial_id]["fixations"]   -> DataFrame (may be empty)
        data[trial_id]["eyetracker"]  -> DataFrame (raw gaze)
    """
    sub_proc = processed_dir / f"sub-{subject_id}"
    sub_int  = intermediate_dir / f"sub-{subject_id}"

    if not sub_int.exists():
        return {}

    data = {}
    for trial_dir in sorted(sub_int.iterdir()):
        if not trial_dir.is_dir():
            continue

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
# Inter-trial fixations
# ---------------------------------------------------------------------------

def extract_intertrial_fixations(
    trials_ordered: list[str],
    data: dict[str, dict],
    eye_cfg: dict,
) -> pd.DataFrame:
    """
    Find eye samples that fall in the gap between consecutive trials and run
    fixation detection on them.

    Each trial's eyetracker data covers [t_start, t_end] of that trial.
    The gap is (t_end_of_trial_N, t_start_of_trial_N+1) in raw timestamps.

    Returns a DataFrame with columns:
        trial_before, trial_after, start_ms, end_ms, duration_ms, x, y,
        gap_start_s, gap_end_s, gap_duration_s
    (x, y are in pixels)
    """
    time_col = eye_cfg["time_col"]

    rows = []

    for i in range(len(trials_ordered) - 1):
        t_before = trials_ordered[i]
        t_after  = trials_ordered[i + 1]

        eye_before = data[t_before]["eyetracker"]
        eye_after  = data[t_after]["eyetracker"]

        if eye_before.empty or eye_after.empty:
            continue

        gap_start = eye_before[time_col].iloc[-1]   # end of trial N
        gap_end   = eye_after[time_col].iloc[0]     # start of trial N+1
        gap_dur   = gap_end - gap_start

        if gap_dur <= 0:
            continue

        # eye samples strictly between the two trials (neither trial has them,
        # so we approximate from the boundary timestamps stored in each trial)
        # Build a synthetic eye DataFrame from the last sample of trial N and
        # the first sample of trial N+1 as anchors, then report the gap.
        # If the experiment recorded a continuous eye stream, the gap samples
        # are not available in the split intermediate files; we still report
        # the gap metadata and the boundary gaze positions.
        gap_row = {
            "trial_before":    t_before,
            "trial_after":     t_after,
            "gap_start_s":     float(gap_start),
            "gap_end_s":       float(gap_end),
            "gap_duration_s":  float(gap_dur),
            "n_fixations":     0,
            "note": "eye samples in gap not available in split intermediate files",
        }
        rows.append(gap_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(subject_id: str,
                    intermediate_dir: Path,
                    processed_dir: Path,
                    aois: list[dict],
                    eye_cfg: dict) -> list[dict]:
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

    # --- Inter-trial (between-games) fixations ---
    trials_ordered = sorted(data.keys())
    gap_df = extract_intertrial_fixations(trials_ordered, data, eye_cfg)
    if not gap_df.empty:
        out_sub = processed_dir / f"sub-{subject_id}"
        out_sub.mkdir(parents=True, exist_ok=True)
        gap_df.to_csv(out_sub / "intertrial_fixations.csv", index=False)
        print(f"         inter-trial gaps saved -> "
              f"data/processed/sub-{subject_id}/intertrial_fixations.csv")

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
            process_subject(sid, intermediate, processed, aois, eye_cfg)
        )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        out_path   = processed / "aoi_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nAOI summary -> {out_path.relative_to(ROOT)}")
        print(summary_df.to_string(index=False))

    print("\nDone.")
