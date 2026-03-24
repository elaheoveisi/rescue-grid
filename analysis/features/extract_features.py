

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
                      maxdist: float = 25.0,
                      mindur: float = 100.0,
                      maxdur: float = 700.0,
                      maxgap: float = 150.0) -> dict:
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
                     minlen: float = 10,
                     maxvel: float = 1000.0,
                     maxdur: float = 150.0) -> dict:
    if sac_df is None or sac_df.empty:
        return {
            "n_saccades":           0,
            "mean_saccade_amp_px":  0.0,
            "total_saccade_dur_ms": 0.0,
        }
    # sac_df = sac_df[
    #     (sac_df["duration_ms"] >= mindur) & (sac_df["duration_ms"] <= maxdur)
    # ]
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


def _saccade_aoi(x: float, y: float, aois: list[dict]) -> str:
    """Return the AOI name containing (x, y), else 'offscreen'."""
    for aoi in aois:
        if aoi["x_min"] <= x <= aoi["x_max"] and aoi["y_min"] <= y <= aoi["y_max"]:
            return aoi["name"]
    return "offscreen"


def quarter_features(game_df:      pd.DataFrame | None,
                     fix_aoi_df:  pd.DataFrame | None,
                     sac_df:      pd.DataFrame | None,
                     eye_df:      pd.DataFrame | None,
                     missing_val: float = 0.0,
                     quarters:    list[int] | None = None,
                     aois:        list[dict] | None = None) -> dict:
    """Per-quarter (25/50/75/100 % of steps) features for all metrics.

    Quarter boundaries are defined by step count (e.g. 0-25% of total steps),
    then converted to timestamps to window fixation, saccade, and eye data.

    Columns produced (one set per quarter q):
        AOI:      q{q}_game_area_pct_dur, q{q}_info_panel_pct_dur, q{q}_chat_panel_pct_dur
        Fixation: q{q}_n_fixations, q{q}_n_fixations_{aoi}, q{q}_mean_fixation_dur_ms, q{q}_total_fixation_dur_ms
        Saccade:  q{q}_n_saccades, q{q}_n_saccades_{aoi}, q{q}_mean_saccade_amp_px, q{q}_total_saccade_dur_ms
        Eye:      q{q}_mean_pupil_diam, q{q}_std_pupil_diam, q{q}_mean_eye_distance, q{q}_pct_missing_eye
        Game:     q{q}_mean_reward, q{q}_n_actions, q{q}_n_llm_calls, q{q}_victims_per_step
    """
    if quarters is None:
        quarters = [25, 50, 75, 100]
    if aois is None:
        aois = []

    aoi_names = [a["name"] for a in aois]

    transition_cols = [f"transitions_{src}_{dst}"
                       for src in aoi_names for dst in aoi_names if src != dst]

    all_cols = (
        ["game_area_pct_dur", "info_panel_pct_dur", "chat_panel_pct_dur"]
        + ["n_fixations"] + [f"n_fixations_{n}" for n in aoi_names]
        + ["mean_fixation_dur_ms", "total_fixation_dur_ms"]
        + ["n_saccades"] + [f"n_saccades_{n}" for n in aoi_names]
        + ["mean_saccade_amp_px", "total_saccade_dur_ms"]
        + ["mean_pupil_diam", "std_pupil_diam", "mean_eye_distance", "pct_missing_eye"]
        + ["mean_reward", "n_actions", "n_llm_calls", "victims_per_step"]
        + transition_cols
    )
    null_row = {f"q{pct}_{col}": None for pct in quarters for col in all_cols}

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

    result       = {}
    prev_ms      = 0.0
    victims_prev = 0
    steps_prev   = 0

    for pct in quarters:
        threshold    = total_steps * pct / 100.0
        at_or_before = game_df[game_df["step_count"] <= threshold]

        if at_or_before.empty:
            for col in all_cols:
                result[f"q{pct}_{col}"] = None
            continue

        # quarter time window (derived from step boundaries)
        t_end_ms    = (float(at_or_before["timestamp"].iloc[-1]) - t0_s) * 1000.0
        t_start_ms  = prev_ms
        t_start_abs = t0_s + t_start_ms / 1000.0
        t_end_abs   = t0_s + t_end_ms   / 1000.0

        # ── game features (slice by timestamp window) ─────────────────────────
        win_game = game_df[
            (game_df["timestamp"] >= t_start_abs) &
            (game_df["timestamp"] <= t_end_abs)
        ]
        if not win_game.empty:
            if "reward" in win_game.columns:
                rewards = pd.to_numeric(win_game["reward"], errors="coerce")
                result[f"q{pct}_mean_reward"] = round(float(rewards.mean()), 4) if not rewards.dropna().empty else None
            else:
                result[f"q{pct}_mean_reward"] = None
            result[f"q{pct}_n_actions"] = (
                int(win_game["action"].notna().sum()) if "action" in win_game.columns else None
            )
            if "llm_response" in win_game.columns:
                result[f"q{pct}_n_llm_calls"] = int(
                    win_game["llm_response"].replace("", pd.NA).notna().sum()
                )
            else:
                result[f"q{pct}_n_llm_calls"] = None
        else:
            result[f"q{pct}_mean_reward"] = None
            result[f"q{pct}_n_actions"]   = None
            result[f"q{pct}_n_llm_calls"] = None

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
            ga_fix = win_fix[win_fix["aoi"] == "game_area"]
            ip_fix = win_fix[win_fix["aoi"] == "info_panel"]
            cp_fix = win_fix[win_fix["aoi"] == "chat_panel"]
            result[f"q{pct}_n_fixations"]              = len(win_fix)
            result[f"q{pct}_n_fixations_game_area"]    = len(ga_fix)
            result[f"q{pct}_n_fixations_info_panel"]   = len(ip_fix)
            result[f"q{pct}_n_fixations_chat_panel"]   = len(cp_fix)
            result[f"q{pct}_mean_fixation_dur_ms"]     = round(float(win_fix["duration_ms"].mean()), 3)
            result[f"q{pct}_total_fixation_dur_ms"]    = round(total_fix_dur, 3)
            if total_fix_dur > 0:
                result[f"q{pct}_game_area_pct_dur"]  = round(float(ga_fix["duration_ms"].sum() / total_fix_dur * 100), 3)
                result[f"q{pct}_info_panel_pct_dur"] = round(float(ip_fix["duration_ms"].sum() / total_fix_dur * 100), 3)
                result[f"q{pct}_chat_panel_pct_dur"] = round(float(cp_fix["duration_ms"].sum() / total_fix_dur * 100), 3)
            else:
                result[f"q{pct}_game_area_pct_dur"]  = None
                result[f"q{pct}_info_panel_pct_dur"] = None
                result[f"q{pct}_chat_panel_pct_dur"] = None
            # ── AOI transitions (from ordered fixation sequence) ──────────────
            if aoi_names and "aoi" in win_fix.columns:
                seq = win_fix[win_fix["aoi"] != "offscreen"]["aoi"].tolist()
                for src in aoi_names:
                    for dst in aoi_names:
                        if src != dst:
                            result[f"q{pct}_transitions_{src}_{dst}"] = sum(
                                1 for a, b in zip(seq[:-1], seq[1:]) if a == src and b == dst
                            )
            else:
                for col in transition_cols:
                    result[f"q{pct}_{col}"] = None
        else:
            result[f"q{pct}_n_fixations"]            = None
            result[f"q{pct}_n_fixations_game_area"]  = None
            result[f"q{pct}_n_fixations_info_panel"] = None
            result[f"q{pct}_n_fixations_chat_panel"] = None
            result[f"q{pct}_mean_fixation_dur_ms"]   = None
            result[f"q{pct}_total_fixation_dur_ms"]  = None
            result[f"q{pct}_game_area_pct_dur"]      = None
            result[f"q{pct}_info_panel_pct_dur"]     = None
            result[f"q{pct}_chat_panel_pct_dur"]     = None
            for col in transition_cols:
                result[f"q{pct}_{col}"] = None

        # ── saccade features ─────────────────────────────────────────────────
        if sac_df is not None and not sac_df.empty and "start_ms" in sac_df.columns:
            win_sac = sac_df[
                (sac_df["start_ms"] >= t_start_ms) &
                (sac_df["start_ms"] <  t_end_ms)
            ]
            if not win_sac.empty:
                result[f"q{pct}_n_saccades"]           = len(win_sac)
                result[f"q{pct}_mean_saccade_amp_px"]  = round(float(win_sac["amplitude"].mean()),  3)
                result[f"q{pct}_total_saccade_dur_ms"] = round(float(win_sac["duration_ms"].sum()), 3)
                if aois and "x_start" in win_sac.columns and "y_start" in win_sac.columns:
                    sac_aoi = win_sac.apply(lambda r: _saccade_aoi(r["x_start"], r["y_start"], aois), axis=1)
                    for n in aoi_names:
                        result[f"q{pct}_n_saccades_{n}"] = int((sac_aoi == n).sum())
                else:
                    for n in aoi_names:
                        result[f"q{pct}_n_saccades_{n}"] = None
            else:
                result[f"q{pct}_n_saccades"]           = None
                result[f"q{pct}_mean_saccade_amp_px"]  = None
                result[f"q{pct}_total_saccade_dur_ms"] = None
                for n in aoi_names:
                    result[f"q{pct}_n_saccades_{n}"] = None
        else:
            result[f"q{pct}_n_saccades"]           = None
            result[f"q{pct}_mean_saccade_amp_px"]  = None
            result[f"q{pct}_total_saccade_dur_ms"] = None
            for n in aoi_names:
                result[f"q{pct}_n_saccades_{n}"] = None

        # ── eye features ─────────────────────────────────────────────────────
        if eye_df is not None and not eye_df.empty and "timestamp" in eye_df.columns:
            win_eye = eye_df[
                (eye_df["timestamp"] >= t_start_abs) &
                (eye_df["timestamp"] <  t_end_abs)
            ]
            if not win_eye.empty:
                if "avg_pupil_diam" in win_eye.columns:
                    valid_pupil = win_eye["avg_pupil_diam"].replace(missing_val, np.nan).dropna()
                    result[f"q{pct}_mean_pupil_diam"] = round(float(valid_pupil.mean()), 4) if not valid_pupil.empty else None
                    result[f"q{pct}_std_pupil_diam"]  = round(float(valid_pupil.std()),  4) if len(valid_pupil) > 1 else None
                else:
                    result[f"q{pct}_mean_pupil_diam"] = None
                    result[f"q{pct}_std_pupil_diam"]  = None
                if "avg_eye_distance" in win_eye.columns:
                    valid_dist = win_eye["avg_eye_distance"].replace(missing_val, np.nan).dropna()
                    result[f"q{pct}_mean_eye_distance"] = round(float(valid_dist.mean()), 4) if not valid_dist.empty else None
                else:
                    result[f"q{pct}_mean_eye_distance"] = None
                if "avg_gaze_point_x" in win_eye.columns:
                    n_total   = len(win_eye)
                    n_missing = int((win_eye["avg_gaze_point_x"].isna() | (win_eye["avg_gaze_point_x"] == missing_val)).sum())
                    result[f"q{pct}_pct_missing_eye"] = round(n_missing / n_total * 100, 2) if n_total > 0 else 0.0
                else:
                    result[f"q{pct}_pct_missing_eye"] = None
            else:
                result[f"q{pct}_mean_pupil_diam"]   = None
                result[f"q{pct}_std_pupil_diam"]    = None
                result[f"q{pct}_mean_eye_distance"] = None
                result[f"q{pct}_pct_missing_eye"]   = None
        else:
            result[f"q{pct}_mean_pupil_diam"]   = None
            result[f"q{pct}_std_pupil_diam"]    = None
            result[f"q{pct}_mean_eye_distance"] = None
            result[f"q{pct}_pct_missing_eye"]   = None

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
                    fix_maxdist: float = 25.0,
                    fix_mindur: float = 100.0,
                    fix_maxdur: float = 700.0,
                    fix_maxgap: float = 150.0,
                    sac_minlen: float = 10.0,
                    sac_maxvel: float = 1000.0,
                    sac_maxdur: float = 150.0,
                    aois: list[dict] | None = None) -> list[dict]:
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

        feat: dict = {
            "subject": subject_id, "trial": trial_id,
            "base_trial": base_trial, "run": run, "expertise": expertise,
            "fix_maxdist": fix_maxdist, "fix_mindur": fix_mindur,
            "fix_maxdur": fix_maxdur,  "fix_maxgap": fix_maxgap,
            "sac_minlen": sac_minlen,  "sac_maxvel": sac_maxvel,
            "sac_maxdur": sac_maxdur,
        }
        feat.update(game_features(streams["game"]))
        feat.update(checkpoint_features(streams["game"], checkpoints or []))
        feat.update(fixation_features(streams["fixations"], maxdist=fix_maxdist, mindur=fix_mindur, maxdur=fix_maxdur, maxgap=fix_maxgap))
        feat.update(saccade_features(streams["saccades"], minlen=sac_minlen, maxvel=sac_maxvel, maxdur=sac_maxdur))
        feat.update(eye_features(streams["eyetracker"], missing_val))
        feat.update(aoi_features(subject_id, trial_id, processed_dir))
        feat.update(quarter_features(
            streams["game"], streams["fixations_aoi"], streams["saccades"],
            streams["eyetracker"], missing_val=missing_val,
            quarters=checkpoints or [25, 50, 75, 100],
            aois=aois or [],
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
    aois             = cfg["aoi"]
    checkpoints      = cfg["checkpoints"]
    fix_cfg          = cfg["fixation"]
    fix_maxdist      = fix_cfg["maxdist"]
    fix_mindur       = fix_cfg["mindur"]
    sac_cfg          = cfg["saccade"]
    sac_minlen       = sac_cfg["minlen"]
    sac_maxvel       = sac_cfg["maxvel"]
    sac_maxdur       = sac_cfg["maxdur"]

    print(f"Extracting features for {len(subjects)} subject(s)\n")

    all_rows = []
    for sid in subjects:
        all_rows.extend(
            process_subject(sid, intermediate_dir, processed_dir, missing_val,
                            expertise=expertise_map.get(sid, "unknown"),
                            checkpoints=checkpoints,
                            fix_maxdist=fix_maxdist,
                            fix_mindur=fix_mindur,
                            sac_minlen=sac_minlen,
                            sac_maxvel=sac_maxvel,
                            sac_maxdur=sac_maxdur,
                            aois=aois)
        )

    if all_rows:
        features_df = pd.DataFrame(all_rows)
        # main features CSV (all columns)
        out_path = processed_dir / "features.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out_path, index=False)
        print(f"\nFeatures      -> {out_path.relative_to(ROOT)}  ({len(features_df)} rows x {len(features_df.columns)} cols)")

        # quarter features CSV (id columns + q* columns only)
        id_cols = ["subject", "trial", "base_trial", "run", "expertise"]
        q_cols  = [c for c in features_df.columns if c.startswith("q") and "_" in c and c[1:3].isdigit()]
        q_out   = processed_dir / "features_quarterly.csv"
        features_df[id_cols + q_cols].to_csv(q_out, index=False)
        print(f"Quarter feats -> {q_out.relative_to(ROOT)}  ({len(features_df)} rows x {len(id_cols) + len(q_cols)} cols)")
    else:
        print("\nNo features extracted.")

    print("\nDone.")
