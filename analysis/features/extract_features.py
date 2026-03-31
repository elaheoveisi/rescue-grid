import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(ROOT, 'data', 'processed')
DATA_INTERMEDIATE = os.path.join(ROOT, 'data', 'intermediate')
PARTICIPANTS = [f'sub-P{str(i).zfill(3)}' for i in range(1, 11)]
"""generates numbers from 1 to 10, formats them as three-digit strings (e.g., '001', '002', ..., '010'), and then creates participant IDs by prefixing each formatted number with 'sub-P' (e.g., 'sub-P001', 'sub-P002', ..., 'sub-P010')."""
CATEGORIES = ['gemini', 'openai', 'dummy']

def infer_category(trial_id):
	"""Infer category from trial directory name."""
	for cat in CATEGORIES:
		if cat in trial_id:
			return cat
	return 'unknown'

def get_all_trial_ids(participant):
	"""Return all trial directory names for a participant under data/processed."""
	sub_dir = os.path.join(DATA_PROCESSED, participant)
	if not os.path.isdir(sub_dir):
		return []
	return [d for d in sorted(os.listdir(sub_dir)) if os.path.isdir(os.path.join(sub_dir, d))]

def get_trial_paths(participant, trial_id):
	trial_dir = os.path.join(DATA_PROCESSED, participant, trial_id)
	interm_dir = os.path.join(DATA_INTERMEDIATE, participant, trial_id)
	return {
		'fixations': os.path.join(trial_dir, 'fixations.csv'),
		'fixations_aoi': os.path.join(trial_dir, 'fixations_aoi.csv'),
		'aoi_transitions': os.path.join(trial_dir, 'aoi_transitions.csv'),
		'game': os.path.join(interm_dir, 'game.csv'),
	}

def extract_fixation_features(fix_df, fix_aoi_df):
	features = {}
	features['n_fixations'] = len(fix_df)
	features['mean_fixation_dur_ms'] = fix_df['duration_ms'].mean()
	features['total_fixation_dur_ms'] = fix_df['duration_ms'].sum()
	if 'aoi' in fix_aoi_df.columns:
		aoi_dur = fix_aoi_df.groupby('aoi')['duration_ms'].sum()
		total = aoi_dur.sum()
		for aoi, dur in aoi_dur.items():
			features[f'{aoi}_pct_dur'] = dur / total if total > 0 else 0
			features[f'n_fixations_{aoi}'] = (fix_aoi_df['aoi'] == aoi).sum()
	return features

def extract_transition_features(trans_df):
	features = {}
	if trans_df.shape[0] > 0:
		src_aois = trans_df.columns[1:]
		for i, src in enumerate(src_aois):
			for j, dst in enumerate(src_aois): #enumerate(src_aois) returns pairs of (index, value) for each element in src_aois
				val = trans_df.iloc[i, j+1]  # j is the index (0, 1, 2, ...).# dst is the value at that index in src_aois.
				features[f'transitions_{src}_{dst}'] = val
	return features

def extract_game_features(game_df):
	features = {}
	features['mean_reward'] = game_df['reward'].mean()
	features['n_actions'] = game_df['action'].notna().sum()
	features['n_llm_calls'] = game_df['llm_response'].notna().sum() if 'llm_response' in game_df.columns else 0
	features['victims_per_step'] = (game_df['saved_victims'].max() / game_df['step_count'].max()) if game_df['step_count'].max() else 0
	features['n_victims_saved'] = game_df['saved_victims'].max()
	return features

def extract_saccade_features(participant, trial_id):
	# Read saccades_summary.csv and extract features for the given participant and trial
	saccades_path = os.path.join(DATA_PROCESSED, 'saccades_summary.csv')
	if not os.path.exists(saccades_path):
		return {}
	try:
		sac_df = pd.read_csv(saccades_path)
	except Exception:
		return {}
	# The 'subject' column is like 'P001', but PARTICIPANTS are 'sub-P001'.
	# Remove 'sub-' prefix for matching
	subj = participant.replace('sub-', '')
	row = sac_df[(sac_df['subject'] == subj) & (sac_df['trial'] == trial_id)]
	if row.empty:
		return {}
	row = row.iloc[0]
	features = {
		'n_saccades': row['n_saccades'],
		'saccades_total_duration_ms': row['total_duration_ms'],
		'saccades_mean_duration_ms': row['mean_duration_ms'],
		'saccades_mean_amplitude_px': row['mean_amplitude_px'],
	}
	return features

def extract_features_for_trial(participant, trial_id):
	paths = get_trial_paths(participant, trial_id)
	try:
		fix_df = pd.read_csv(paths['fixations'])
		fix_aoi_df = pd.read_csv(paths['fixations_aoi'])
		trans_df = pd.read_csv(paths['aoi_transitions'])
		game_df = pd.read_csv(paths['game'])
	except Exception as e:
		print(f"Skipping {participant} {trial_id}: {e}")
		return None
	category = infer_category(trial_id)
	features = {'participant': participant, 'trial': trial_id, 'category': category}
	features.update(extract_fixation_features(fix_df, fix_aoi_df))
	features.update(extract_transition_features(trans_df))
	features.update(extract_game_features(game_df))
	features.update(extract_saccade_features(participant, trial_id))
	return features

def main():
	all_features = []
	best_features = []
	for participant in PARTICIPANTS:
		for trial_id in get_all_trial_ids(participant):
			feats = extract_features_for_trial(participant, trial_id)
			if feats:
				all_features.append(feats)
				if trial_id.endswith('_best'):
					best_features.append(feats)

	all_out  = os.path.join(DATA_PROCESSED, 'all_features.csv')
	best_out = os.path.join(DATA_PROCESSED, 'best_features.csv')

	pd.DataFrame(all_features).to_csv(all_out, index=False)
	print(f'All trials  -> {all_out}')

	pd.DataFrame(best_features).to_csv(best_out, index=False)
	print(f'Best trials -> {best_out}')

if __name__ == '__main__':
	main()
