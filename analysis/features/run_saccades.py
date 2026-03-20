
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "igaze" / "igaze"))

from detectors import saccade_detection  # noqa: E402

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
# Run saccades
# ---------------------------------------------------------------------------

def run_saccades(cfg: dict) -> None:
    subjects     = [str(s) for s in cfg.get("sub", [])]
    intermediate = ROOT / cfg["paths"]["intermediate"]
    processed    = ROOT / cfg["paths"]["processed"]
    eye_cfg      = cfg.get("eyetracker", {})
    sac_cfg      = cfg.get("saccade", {})

    x_col    = eye_cfg["x_col"]
    y_col    = eye_cfg["y_col"]
    time_col = eye_cfg["time_col"]
    screen_w = eye_cfg["screen_w"]
    screen_h = eye_cfg["screen_h"]
    missing  = eye_cfg["missing"]
    minlen   = sac_cfg["minlen"]
    maxvel   = sac_cfg["maxvel"]
    maxacc   = sac_cfg["maxacc"]
    maxgap   = sac_cfg.get("maxgap", 200)

    summaries = []

    for sid in subjects:
        data = load_subject_trials(sid, intermediate)

        if not data:
            print(f"[SKIP] {sid}: no trial data in {intermediate.relative_to(ROOT)}")
            continue

        print(f"[SUB]  {sid}")

        for trial_id, streams in data.items():
            eye_df = streams["eyetracker"].copy()

            if eye_df.empty:
                print(f"         {trial_id:35s}  empty eyetracker — skipping")
                continue

            # convert timestamps to relative ms
            eye_df[time_col] = (eye_df[time_col] - eye_df[time_col].iloc[0]) * 1000.0

            # scale normalized gaze (0-1) to pixels
            eye_df[x_col] = eye_df[x_col] * screen_w
            eye_df[y_col] = eye_df[y_col] * screen_h

            x    = eye_df[x_col].to_numpy(dtype=float)
            y    = eye_df[y_col].to_numpy(dtype=float)
            time = eye_df[time_col].to_numpy(dtype=float)

            _, end_saccades = saccade_detection(
                x, y, time,
                missing=missing,
                minlen=minlen,
                maxvel=maxvel,
                maxacc=maxacc,
                maxgap=maxgap,
            )

            rows = []
            for sac_id, sac in enumerate(end_saccades, start=1):
                start_t, end_t, duration, xs, ys, xe, ye = sac
                amplitude = float(((xe - xs) ** 2 + (ye - ys) ** 2) ** 0.5)
                rows.append({
                    "saccade_id":  sac_id,
                    "start_ms":    float(start_t),
                    "end_ms":      float(end_t),
                    "duration_ms": float(duration),
                    "x_start":     float(xs),
                    "y_start":     float(ys),
                    "x_end":       float(xe),
                    "y_end":       float(ye),
                    "amplitude":   amplitude,
                })

            n        = len(rows)
            total_ms = sum(r["duration_ms"] for r in rows)
            mean_dur = total_ms / n if n > 0 else 0.0
            mean_amp = float(np.mean([r["amplitude"] for r in rows])) if rows else 0.0

            print(f"         {trial_id:35s}  "
                  f"saccades={n:>4}  "
                  f"mean_dur={mean_dur:>6.1f}ms  "
                  f"mean_amp={mean_amp:>6.1f}px")

            # save
            out_dir = processed / f"sub-{sid}" / trial_id
            out_dir.mkdir(parents=True, exist_ok=True)

            if rows:
                sac_df = pd.DataFrame(rows)
                sac_df.to_csv(out_dir / "saccades.csv", index=False)
                sac_df.to_hdf(out_dir / "saccades.h5", key="saccades", mode="w")

            summaries.append({
                "subject":           sid,
                "trial":             trial_id,
                "minlen":            minlen,
                "maxvel":            maxvel,
                "maxacc":            maxacc,
                "n_saccades":        n,
                "total_duration_ms": total_ms,
                "mean_duration_ms":  mean_dur,
                "mean_amplitude_px": mean_amp,
            })

    if summaries:
        summary_df = pd.DataFrame(summaries)
        out_path   = processed / "saccades_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nSummary -> {out_path.relative_to(ROOT)}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo saccades computed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    run_saccades(cfg)
