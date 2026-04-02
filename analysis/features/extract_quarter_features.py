import os
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED   = os.path.join(_HERE, '..', '..', 'data', 'processed')
DATA_INTERMEDIATE = os.path.join(_HERE, '..', '..', 'data', 'intermediate')

PARTICIPANTS = [f'sub-P{str(i).zfill(3)}' for i in range(1, 11)]
CATEGORIES   = ['gemini', 'openai', 'dummy']
TRIAL_SUFFIX = {
    'gemini': 'trial_detailed_gemini_best',
    'openai': 'trial_detailed_openai_best',
    'dummy':  'trial_dummy_best',
}
QUARTERS = [25, 50, 75, 100]


def get_step_cutoff_ms(game_df, eye_t0, pct):
    """Return the relative-ms timestamp corresponding to pct% of total steps."""
    total_steps = game_df['step_count'].max()
    threshold   = total_steps * pct / 100.0
    # first row where step_count reaches the threshold
    rows = game_df[game_df['step_count'] >= threshold]
    if rows.empty:
        rows = game_df  # fall back to last row
    game_ts = rows.iloc[0]['timestamp']
    return (game_ts - eye_t0) * 1000.0  # convert to relative ms


def fixation_features_up_to(fix_aoi_df, cutoff_ms):
    df = fix_aoi_df[fix_aoi_df['start_ms'] <= cutoff_ms]
    features = {}
    features['n_fixations'] = len(df)
    features['mean_fixation_dur_ms']  = df['duration_ms'].mean() if len(df) else 0.0
    features['total_fixation_dur_ms'] = df['duration_ms'].sum()
    if 'aoi' in df.columns and len(df):
        aoi_dur = df.groupby('aoi')['duration_ms'].sum()
        total   = aoi_dur.sum()
        for aoi, dur in aoi_dur.items():
            features[f'{aoi}_pct_dur']     = dur / total if total > 0 else 0.0
            features[f'n_fixations_{aoi}'] = int((df['aoi'] == aoi).sum())
    return features


def saccade_features_up_to(sac_df, cutoff_ms):
    df = sac_df[sac_df['start_ms'] <= cutoff_ms]
    features = {}
    features['n_saccades']           = len(df)
    features['mean_saccade_amp_px']  = float(df['amplitude'].mean()) if len(df) else 0.0
    features['total_saccade_dur_ms'] = df['duration_ms'].sum()
    return features


def transition_features_up_to(fix_aoi_df, cutoff_ms):
    df   = fix_aoi_df[fix_aoi_df['start_ms'] <= cutoff_ms].reset_index(drop=True)
    aois = ['game_area', 'info_panel', 'chat_panel']
    features = {f'transitions_{src}_{dst}': 0 for src in aois for dst in aois}
    for i in range(len(df) - 1):
        src, dst = df.loc[i, 'aoi'], df.loc[i + 1, 'aoi']
        key = f'transitions_{src}_{dst}'
        if key in features:
            features[key] += 1
    return features


def pupil_features_up_to(eye_df, eye_t0, cutoff_ms):
    cutoff_ts = eye_t0 + cutoff_ms / 1000.0
    df = eye_df[eye_df['timestamp'] <= cutoff_ts]
    pupil = df['avg_pupil_diam'].replace(0, np.nan).dropna()
    return {'std_pupil_diam': float(pupil.std()) if len(pupil) > 1 else 0.0}


def game_features_up_to(game_df, pct):
    total_steps = game_df['step_count'].max()
    threshold   = total_steps * pct / 100.0
    df = game_df[game_df['step_count'] <= threshold]
    features = {}
    features['saved_victims']   = int(df['saved_victims'].max()) if len(df) else 0
    features['victims_per_step'] = (
        df['saved_victims'].max() / df['step_count'].max()
        if len(df) and df['step_count'].max() > 0 else 0.0
    )
    features['mean_reward']  = df['reward'].mean() if len(df) else 0.0
    features['n_actions']    = int(df['action'].notna().sum())
    features['n_llm_calls']  = int(df['llm_response'].notna().sum()) if 'llm_response' in df.columns else 0
    return features


def extract_quarter_features(participant, category, pct):
    trial_id  = TRIAL_SUFFIX[category]
    proc_dir  = os.path.join(DATA_PROCESSED,    participant, trial_id)
    interm_dir = os.path.join(DATA_INTERMEDIATE, participant, trial_id)

    try:
        fix_aoi_df = pd.read_csv(os.path.join(proc_dir,   'fixations_aoi.csv'))
        sac_df     = pd.read_csv(os.path.join(proc_dir,   'saccades.csv'))
        game_df    = pd.read_csv(os.path.join(interm_dir, 'game.csv'))
        eye_df     = pd.read_hdf(os.path.join(interm_dir, 'eyetracker.h5'), key='eyetracker')
    except Exception as e:
        print(f"Skipping {participant} {category} q{pct}: {e}")
        return None

    eye_t0    = eye_df['timestamp'].iloc[0]
    cutoff_ms = get_step_cutoff_ms(game_df, eye_t0, pct)

    features = {
        'participant': participant,
        'category':    category,
        'quarter_pct': pct,
    }
    features.update(game_features_up_to(game_df, pct))
    features.update(fixation_features_up_to(fix_aoi_df, cutoff_ms))
    features.update(saccade_features_up_to(sac_df, cutoff_ms))
    features.update(transition_features_up_to(fix_aoi_df, cutoff_ms))
    features.update(pupil_features_up_to(eye_df, eye_t0, cutoff_ms))
    return features


def main():
    output_dir = os.path.join(_HERE, '..', '..', 'data', 'processed')

    for pct in QUARTERS:
        rows = []
        for participant in PARTICIPANTS:
            for category in CATEGORIES:
                feats = extract_quarter_features(participant, category, pct)
                if feats:
                    rows.append(feats)
        df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f'features_q{pct}.csv')
        df.to_csv(out_path, index=False)
        print(f'Q{pct:3d}% -> {out_path}  ({len(df)} rows)')


if __name__ == '__main__':
    main()
