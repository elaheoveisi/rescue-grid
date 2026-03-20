

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT   = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "configs" / "config_analysis.yml"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_h5(path: Path, key: str) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_hdf(path, key=key)
    return None


def load_trial_data(subject_id: str,
                    trial_id: str,
                    intermediate_dir: Path,
                    processed_dir: Path) -> dict[str, pd.DataFrame | None]:
    base_int  = intermediate_dir / f"sub-{subject_id}" / trial_id
    base_proc = processed_dir    / f"sub-{subject_id}" / trial_id
    return {
        "game":       _read_h5(base_int  / "game.h5",       "game"),
        "eyetracker": _read_h5(base_int  / "eyetracker.h5", "eyetracker"),
        "fixations":  _read_h5(base_proc / "fixations.h5",  "fixations"),
        "saccades":   _read_h5(base_proc / "saccades.h5",   "saccades"),
    }


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def game_features(game_df: pd.DataFrame) -> dict:
    if game_df is None or game_df.empty:
        return {}

    last = game_df.iloc[-1]

    # mission outcome from terminated / truncated flags
    outcome = "ongoing"
    if "terminated" in game_df.columns and game_df["terminated"].any():
        outcome = "completed"
    elif "truncated" in game_df.columns and game_df["truncated"].any():
        outcome = "truncated"
    elif "mission_status" in game_df.columns:
        status_vals = game_df["mission_status"].dropna().unique()
        if "success" in status_vals:
            outcome = "completed"
        elif "timeout" in status_vals or "failed" in status_vals:
            outcome = "timeout"

    step_count = int(last.get("step_count", 0))   if "step_count" in last.index  else 0
    max_steps  = int(last.get("max_steps", 1))    if "max_steps"  in last.index  else 1
    total_steps = (
        int(game_df["total_steps"].dropna().iloc[-1])
        if "total_steps" in game_df.columns and game_df["total_steps"].notna().any()
        else step_count
    )

    rewards = pd.to_numeric(game_df["reward"], errors="coerce") if "reward" in game_df.columns else pd.Series(dtype=float)

    n_actions = (
        game_df["action"].notna().sum()
        if "action" in game_df.columns else 0
    )

    # count rows where the LLM was actually queried (non-null, non-empty response)
    n_llm_calls = 0
    if "llm_response" in game_df.columns:
        n_llm_calls = int(
            game_df["llm_response"]
            .replace("", pd.NA)
            .notna()
            .sum()
        )

    duration_s = float(game_df["timestamp"].iloc[-1] - game_df["timestamp"].iloc[0])

    return {
        "final_saved_victims":     int(last.get("saved_victims", 0))     if "saved_victims"     in last.index else None,
        "final_remaining_victims": int(last.get("remaining_victims", 0)) if "remaining_victims" in last.index else None,
        "final_step_count":        step_count,
        "total_steps":             total_steps,
        "max_steps":               max_steps,
        "steps_used_pct":          round(step_count / max_steps * 100, 2) if max_steps > 0 else None,
        "mission_outcome":         outcome,
        "total_reward":            float(rewards.sum()),
        "mean_reward":             float(rewards.mean()) if not rewards.empty else 0.0,
        "n_actions":               int(n_actions),
        "n_llm_calls":             n_llm_calls,
        "prompt_type":             str(last.get("prompt_type", "")) if "prompt_type" in last.index else None,
        "llm_model":               str(last.get("llm_model",   "")) if "llm_model"   in last.index else None,
        "llm_provider":            str(last.get("llm_provider","")) if "llm_provider" in last.index else None,
        "trial_duration_s":        round(duration_s, 4),
    }


def aoi_features(subject_id: str, trial_id: str, processed_dir: Path) -> dict:
    """Read AOI stats for this trial from aoi_summary.csv (written by AOI_fixation.py)."""
    summary_path = processed_dir / "aoi_summary.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    row = summary[(summary["subject"] == subject_id) & (summary["trial"] == trial_id)]
    if row.empty:
        return {}
    exclude = {"subject", "trial", "n_fixations", "n_fixations_offscreen"}
    return {c: row.iloc[0][c] for c in row.columns if c not in exclude}


def checkpoint_features(game_df: pd.DataFrame | None, checkpoints: list[int]) -> dict:
    """Victims saved at each step checkpoint. If the trial ended before the checkpoint, use the last recorded value."""
    if game_df is None or game_df.empty or "step_count" not in game_df.columns or "saved_victims" not in game_df.columns:
        return {f"victims_at_step_{s}": None for s in checkpoints}

    game_df = game_df.copy()
    game_df["step_count"]    = pd.to_numeric(game_df["step_count"],    errors="coerce")
    game_df["saved_victims"] = pd.to_numeric(game_df["saved_victims"], errors="coerce")

    result = {}
    for step in checkpoints:
        at_or_before = game_df[game_df["step_count"] <= step]
        if at_or_before.empty:
            result[f"victims_at_step_{step}"] = None
        else:
            result[f"victims_at_step_{step}"] = int(at_or_before["saved_victims"].iloc[-1])
    return result


def fixation_features(fix_df: pd.DataFrame | None) -> dict:
    if fix_df is None or fix_df.empty:
        return {
            "n_fixations":          0,
            "mean_fixation_dur_ms": 0.0,
            "total_fixation_dur_ms": 0.0,
        }
    return {
        "n_fixations":           len(fix_df),
        "mean_fixation_dur_ms":  round(float(fix_df["duration_ms"].mean()), 3),
        "total_fixation_dur_ms": round(float(fix_df["duration_ms"].sum()),  3),
    }


def saccade_features(sac_df: pd.DataFrame | None) -> dict:
    if sac_df is None or sac_df.empty:
        return {
            "n_saccades":           0,
            "mean_saccade_amp_px":  0.0,
            "total_saccade_dur_ms": 0.0,
        }
    return {
        "n_saccades":           len(sac_df),
        "mean_saccade_amp_px":  round(float(sac_df["amplitude"].mean()),    3),
        "total_saccade_dur_ms": round(float(sac_df["duration_ms"].sum()),   3),
    }


def eye_features(eye_df: pd.DataFrame | None, missing_val: float = 0.0) -> dict:
    if eye_df is None or eye_df.empty:
        return {}

    feats: dict = {}

    if "avg_pupil_diam" in eye_df.columns:
        valid_pupil = eye_df["avg_pupil_diam"].replace(missing_val, np.nan).dropna()
        feats["mean_pupil_diam"] = round(float(valid_pupil.mean()), 4) if not valid_pupil.empty else None

    if "avg_eye_distance" in eye_df.columns:
        valid_dist = eye_df["avg_eye_distance"].replace(missing_val, np.nan).dropna()
        feats["mean_eye_distance"] = round(float(valid_dist.mean()), 4) if not valid_dist.empty else None

    if "avg_gaze_point_x" in eye_df.columns:
        n_total   = len(eye_df)
        n_missing = int((eye_df["avg_gaze_point_x"] == missing_val).sum())
        feats["n_missing_eye"]  = n_missing
        feats["pct_missing_eye"] = round(n_missing / n_total * 100, 2) if n_total > 0 else 0.0

    return feats


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(subject_id: str,
                    intermediate_dir: Path,
                    processed_dir: Path,
                    missing_val: float,
                    expertise: str = "unknown",
                    checkpoints: list[int] | None = None) -> list[dict]:
    sub_int = intermediate_dir / f"sub-{subject_id}"
    if not sub_int.exists():
        print(f"  [SKIP] {subject_id}: no intermediate data")
        return []

    print(f"  [SUB]  {subject_id}")
    rows = []

    for trial_dir in sorted(sub_int.iterdir()):
        if not trial_dir.is_dir():
            continue

        trial_id = trial_dir.name
        m = re.match(r'^(.+)_run(\d+)$', trial_id)
        base_trial = m.group(1) if m else trial_id
        run        = int(m.group(2)) if m else 1

        streams  = load_trial_data(subject_id, trial_id, intermediate_dir, processed_dir)

        feat: dict = {"subject": subject_id, "trial": trial_id,
                      "base_trial": base_trial, "run": run, "expertise": expertise}
        feat.update(game_features(streams["game"]))
        feat.update(checkpoint_features(streams["game"], checkpoints or []))
        feat.update(fixation_features(streams["fixations"]))
        feat.update(saccade_features(streams["saccades"]))
        feat.update(eye_features(streams["eyetracker"], missing_val))
        feat.update(aoi_features(subject_id, trial_id, processed_dir))

        rows.append(feat)

        print(f"         {trial_id:35s}  "
              f"saved={feat.get('final_saved_victims', '?'):>3}  "
              f"steps={feat.get('final_step_count', '?'):>4}  "
              f"llm_calls={feat.get('n_llm_calls', 0):>4}  "
              f"fixations={feat.get('n_fixations', 0):>4}  "
              f"saccades={feat.get('n_saccades', 0):>4}  "
              f"outcome={feat.get('mission_outcome', '?')}")

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    subjects         = [str(s) for s in cfg.get("sub", [])]
    intermediate_dir = ROOT / cfg["paths"]["intermediate"]
    processed_dir    = ROOT / cfg["paths"]["processed"]
    missing_val      = cfg.get("eyetracker", {}).get("missing", 0.0)
    expertise_map    = {str(k): str(v) for k, v in cfg.get("expertise", {}).items()}

    print(f"Extracting features for {len(subjects)} subject(s)\n")

    all_rows = []
    for sid in subjects:
        all_rows.extend(
            process_subject(sid, intermediate_dir, processed_dir, missing_val,
                            expertise=expertise_map.get(sid, "unknown"))
        )

    if all_rows:
        features_df = pd.DataFrame(all_rows)
        out_path    = processed_dir / "features.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out_path, index=False)
        print(f"\nFeatures -> {out_path.relative_to(ROOT)}  ({len(features_df)} rows x {len(features_df.columns)} cols)")
    else:
        print("\nNo features extracted.")

    print("\nDone.")
