

from pathlib import Path

import pandas as pd
import yaml
from igaze.detectors import fixation_detection


def fixation_summary(Efix: list) -> dict:
    if not Efix:
        return {"count": 0, "mean_duration_ms": 0.0, "total_duration_ms": 0.0}
    durations = [f[2] for f in Efix]
    return {
        "count":            len(durations),
        "mean_duration_ms": sum(durations) / len(durations),
        "total_duration_ms": sum(durations),
    }

ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG = ROOT / "configs" / "config_analysis.yml"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_subject_trials(subject_id: str, intermediate_dir: Path) -> dict[str, dict]:
    """Load all trial eyetracker.h5 files for a subject.

    Returns:
        data[trial_id]["eyetracker"] -> DataFrame
    """
    sub_dir = intermediate_dir / f"sub-{subject_id}"
    if not sub_dir.exists():
        return {}

    data = {}
    for trial_dir in sorted(sub_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        h5 = trial_dir / "eyetracker.h5"
        if not h5.exists():
            continue
        data[trial_dir.name] = {
            "eyetracker": pd.read_hdf(h5, key="eyetracker")
        }
    return data


# ---------------------------------------------------------------------------
# Run fixations
# ---------------------------------------------------------------------------

def run_fixations(cfg: dict) -> None:
    subjects        = [str(s) for s in cfg.get("sub", [])]
    intermediate    = ROOT / cfg["paths"]["intermediate"]
    processed       = ROOT / cfg["paths"]["processed"]
    eye_cfg         = cfg.get("eyetracker", {})
    fix_cfg         = cfg.get("fixation", {})

    x_col    = eye_cfg["x_col"]
    y_col    = eye_cfg["y_col"]
    time_col = eye_cfg["time_col"]
    screen_w = eye_cfg["screen_w"]
    screen_h = eye_cfg["screen_h"]
    missing  = eye_cfg["missing"]
    maxdist  = fix_cfg["maxdist"]
    mindur   = fix_cfg["mindur"]

    summaries      = []
    best_summaries = []

    for sid in subjects:
        data = load_subject_trials(sid, intermediate)

        if not data:
            print(f"[SKIP] P{sid}: no trial data in {intermediate.relative_to(ROOT)}")
            continue

        print(f"[SUB]  {sid}")

        for trial_id, streams in data.items():
            if not trial_id.endswith("_best"):
                continue

            eye_df = streams["eyetracker"].copy()

            if eye_df.empty:
                print(f"         {trial_id:35s}  empty eyetracker — skipping")
                continue

            # convert timestamps to relative ms
            eye_df[time_col] = (eye_df[time_col] - eye_df[time_col].iloc[0]) * 1000.0

            # drop missing gaze samples
            eye_df = eye_df.dropna(subset=[x_col, y_col])

            # scale normalized gaze (0-1) to pixels for distance-based detection
            eye_df[x_col] = eye_df[x_col] * screen_w
            eye_df[y_col] = eye_df[y_col] * screen_h

            # run fixation detection
            _, Efix = fixation_detection(
                x=eye_df[x_col].to_numpy(),
                y=eye_df[y_col].to_numpy(),
                time=eye_df[time_col].to_numpy(),
                missing=missing,
                maxdist=maxdist,
                mindur=mindur,
            )
            summary = fixation_summary(Efix)

            print(f"         {trial_id:35s}  "
                  f"fixations={summary['count']:>4}  "
                  f"mean_dur={summary['mean_duration_ms']:>7.1f}ms")

            # save
            out_dir = processed / f"sub-{sid}" / trial_id
            out_dir.mkdir(parents=True, exist_ok=True)

            if Efix:
                fix_df = pd.DataFrame(
                    Efix, columns=["start_ms", "end_ms", "duration_ms", "x", "y"]
                )
                fix_df.to_csv(out_dir / "fixations.csv", index=False)
                fix_df.to_hdf(out_dir / "fixations.h5", key="fixations", mode="w")
                streams["fixations"] = fix_df

            row = {
                "subject": sid, "trial": trial_id,
                "maxdist": maxdist, "mindur": mindur,
                **summary,
            }
            summaries.append(row)
            if trial_id.endswith("_best"):
                best_summaries.append(row)

    # save overall summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        out_path   = processed / "fixations_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nSummary          -> {out_path.relative_to(ROOT)}")

    # save best-run summary
    if best_summaries:
        best_df  = pd.DataFrame(best_summaries)
        best_out = processed / "fixations_best_run_summary.csv"
        best_df.to_csv(best_out, index=False)
        print(f"Best-run summary -> {best_out.relative_to(ROOT)}")
        print(best_df.to_string(index=False))
    else:
        print("\nNo fixations computed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    run_fixations(cfg)
