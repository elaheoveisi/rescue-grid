

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
        "game":          _read_h5(base_int  / "game.h5",          "game"),
        "eyetracker":    _read_h5(base_int  / "eyetracker.h5",    "eyetracker"),
        "fixations":     _read_h5(base_proc / "fixations.h5",     "fixations"),
        "fixations_aoi": _read_h5(base_proc / "fixations_aoi.h5", "fixations_aoi"),
        "saccades":      _read_h5(base_proc / "saccades.h5",      "saccades"),
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
    """Victims saved at each percentage checkpoint of the trial's total steps.

    checkpoints: list of percentages, e.g. [25, 50, 75, 100].
    The actual step threshold is computed as total_steps * pct / 100 per trial.
    """
    if game_df is None or game_df.empty or "step_count" not in game_df.columns or "saved_victims" not in game_df.columns:
        return {f"victims_at_pct_{p}": None for p in checkpoints}

    game_df = game_df.copy()
    game_df["step_count"]    = pd.to_numeric(game_df["step_count"],    errors="coerce")
    game_df["saved_victims"] = pd.to_numeric(game_df["saved_victims"], errors="coerce")

    total_steps = game_df["step_count"].max()
    if pd.isna(total_steps) or total_steps == 0:
        return {f"victims_at_pct_{p}": None for p in checkpoints}

    result = {}
    for pct in checkpoints:
        threshold = total_steps * pct / 100.0
        at_or_before = game_df[game_df["step_count"] <= threshold]
        if at_or_before.empty:
            result[f"victims_at_pct_{pct}"] = None
        else:
            result[f"victims_at_pct_{pct}"] = int(at_or_before["saved_victims"].iloc[-1])
    return result


def fixation_features(fix_df: pd.DataFrame | None,
                      mindur: float = 50.0,
                      maxdur: float = 400.0) -> dict:
    if fix_df is None or fix_df.empty:
        return {
            "n_fixations":           0,
            "mean_fixation_dur_ms":  0.0,
            "total_fixation_dur_ms": 0.0,
        }
    fix_df = fix_df[
        (fix_df["duration_ms"] >= mindur) & (fix_df["duration_ms"] <= maxdur)
    ]
    if fix_df.empty:
        return {
            "n_fixations":           0,
            "mean_fixation_dur_ms":  0.0,
            "total_fixation_dur_ms": 0.0,
        }
    return {
        "n_fixations":           len(fix_df),
        "mean_fixation_dur_ms":  round(float(fix_df["duration_ms"].mean()), 3),
        "total_fixation_dur_ms": round(float(fix_df["duration_ms"].sum()),  3),
    }


def saccade_features(sac_df: pd.DataFrame | None,
                     mindur: float = 10.0,
                     maxdur: float = 150.0) -> dict:
    if sac_df is None or sac_df.empty:
        return {
            "n_saccades":           0,
            "mean_saccade_amp_px":  0.0,
            "total_saccade_dur_ms": 0.0,
        }
    sac_df = sac_df[
        (sac_df["duration_ms"] >= mindur) & (sac_df["duration_ms"] <= maxdur)
    ]
    if sac_df.empty:
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


def quarter_features(game_df:      pd.DataFrame | None,
                     fix_aoi_df:  pd.DataFrame | None,
                     sac_df:      pd.DataFrame | None,
                     eye_df:      pd.DataFrame | None,
                     missing_val: float = 0.0,
                     quarters:    list[int] | None = None) -> dict:
    """Per-quarter (25/50/75/100 % of steps) eye-tracking and performance features.

    Columns produced (one set per quarter q):
        q{q}_game_area_pct_dur, q{q}_chat_panel_pct_dur,
        q{q}_mean_fixation_dur_ms, q{q}_mean_saccade_amp_px,
        q{q}_std_pupil_diam, q{q}_victims_per_step
    """
    if quarters is None:
        quarters = [25, 50, 75, 100]

    null_row = {}
    for pct in quarters:
        for col in ["game_area_pct_dur", "chat_panel_pct_dur",
                    "mean_fixation_dur_ms", "mean_saccade_amp_px",
                    "std_pupil_diam", "victims_per_step"]:
            null_row[f"q{pct}_{col}"] = None

    if game_df is None or game_df.empty:
        return null_row
    if "step_count" not in game_df.columns or "timestamp" not in game_df.columns:
        return null_row

    game_df = game_df.copy()
    game_df["step_count"] = pd.to_numeric(game_df["step_count"], errors="coerce")
    if "saved_victims" in game_df.columns:
        game_df["saved_victims"] = pd.to_numeric(game_df["saved_victims"], errors="coerce")

    # reference time = start of active game period
    if "mission_status" in game_df.columns:
        active = game_df[game_df["mission_status"] == "continue"]
        if active.empty:
            active = game_df
    else:
        active = game_df
    t0_s = float(active["timestamp"].iloc[0])

    total_steps = game_df["step_count"].max()
    if pd.isna(total_steps) or total_steps == 0:
        return null_row

    result        = {}
    prev_ms       = 0.0
    victims_prev  = 0
    steps_prev    = 0

    for pct in quarters:
        threshold    = total_steps * pct / 100.0
        at_or_before = game_df[game_df["step_count"] <= threshold]

        if at_or_before.empty:
            for col in ["game_area_pct_dur", "chat_panel_pct_dur",
                        "mean_fixation_dur_ms", "mean_saccade_amp_px",
                        "std_pupil_diam", "victims_per_step"]:
                result[f"q{pct}_{col}"] = None
            continue

        t_end_ms = (float(at_or_before["timestamp"].iloc[-1]) - t0_s) * 1000.0
        t_start_ms = prev_ms

        # ── fixation-based features ──────────────────────────────────────────
        if fix_aoi_df is not None and not fix_aoi_df.empty and "start_ms" in fix_aoi_df.columns:
            win_fix = fix_aoi_df[
                (fix_aoi_df["start_ms"] >= t_start_ms) &
                (fix_aoi_df["start_ms"] <  t_end_ms)
            ]
        else:
            win_fix = pd.DataFrame()

        if not win_fix.empty:
            total_fix_dur = float(win_fix["duration_ms"].sum())
            if total_fix_dur > 0:
                ga_dur = win_fix[win_fix["aoi"] == "game_area"]["duration_ms"].sum()
                cp_dur = win_fix[win_fix["aoi"] == "chat_panel"]["duration_ms"].sum()
                result[f"q{pct}_game_area_pct_dur"]    = round(float(ga_dur / total_fix_dur * 100), 3)
                result[f"q{pct}_chat_panel_pct_dur"]   = round(float(cp_dur / total_fix_dur * 100), 3)
            else:
                result[f"q{pct}_game_area_pct_dur"]    = None
                result[f"q{pct}_chat_panel_pct_dur"]   = None
            result[f"q{pct}_mean_fixation_dur_ms"] = round(float(win_fix["duration_ms"].mean()), 3)
        else:
            result[f"q{pct}_game_area_pct_dur"]    = None
            result[f"q{pct}_chat_panel_pct_dur"]   = None
            result[f"q{pct}_mean_fixation_dur_ms"] = None

        # ── saccade features ─────────────────────────────────────────────────
        if sac_df is not None and not sac_df.empty and "start_ms" in sac_df.columns:
            win_sac = sac_df[
                (sac_df["start_ms"] >= t_start_ms) &
                (sac_df["start_ms"] <  t_end_ms)
            ]
            result[f"q{pct}_mean_saccade_amp_px"] = (
                round(float(win_sac["amplitude"].mean()), 3) if not win_sac.empty else None
            )
        else:
            result[f"q{pct}_mean_saccade_amp_px"] = None

        # ── pupil std ────────────────────────────────────────────────────────
        if (eye_df is not None and not eye_df.empty
                and "avg_pupil_diam" in eye_df.columns
                and "timestamp" in eye_df.columns):
            t_start_abs = t0_s + t_start_ms / 1000.0
            t_end_abs   = t0_s + t_end_ms   / 1000.0
            win_eye = eye_df[
                (eye_df["timestamp"] >= t_start_abs) &
                (eye_df["timestamp"] <  t_end_abs)
            ]
            valid_pupil = win_eye["avg_pupil_diam"].replace(missing_val, np.nan).dropna()
            result[f"q{pct}_std_pupil_diam"] = (
                round(float(valid_pupil.std()), 4) if len(valid_pupil) > 1 else None
            )
        else:
            result[f"q{pct}_std_pupil_diam"] = None

        # ── victims per step ─────────────────────────────────────────────────
        steps_end   = int(at_or_before["step_count"].iloc[-1])
        victims_end = (
            int(at_or_before["saved_victims"].iloc[-1])
            if "saved_victims" in at_or_before.columns
               and not at_or_before["saved_victims"].isna().all()
            else 0
        )
        steps_in_q   = steps_end   - steps_prev
        victims_in_q = victims_end - victims_prev
        result[f"q{pct}_victims_per_step"] = (
            round(victims_in_q / steps_in_q, 6) if steps_in_q > 0 else None
        )

        prev_ms      = t_end_ms
        victims_prev = victims_end
        steps_prev   = steps_end

    return result


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
        is_missing = eye_df["avg_gaze_point_x"].isna() | (eye_df["avg_gaze_point_x"] == missing_val)
        n_missing  = int(is_missing.sum())
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
                    checkpoints: list[int] | None = None,
                    fix_mindur: float = 50.0,
                    fix_maxdur: float = 400.0,
                    sac_mindur: float = 10.0,
                    sac_maxdur: float = 150.0) -> list[dict]:
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
        feat.update(fixation_features(streams["fixations"], mindur=fix_mindur, maxdur=fix_maxdur))
        feat.update(saccade_features(streams["saccades"], mindur=sac_mindur, maxdur=sac_maxdur))
        feat.update(eye_features(streams["eyetracker"], missing_val))
        feat.update(aoi_features(subject_id, trial_id, processed_dir))
        feat.update(quarter_features(
            streams["game"], streams["fixations_aoi"], streams["saccades"],
            streams["eyetracker"], missing_val=missing_val,
            quarters=checkpoints or [25, 50, 75, 100],
        ))

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
