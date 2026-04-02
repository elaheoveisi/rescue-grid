import yaml
from pathlib import Path

from analysis.data.xdf import run_from_config as run_xdf
from analysis.features.run_fixations import run_fixations
from analysis.features.run_saccades import run_saccades
from analysis.glmm import run_from_config
from analysis.glmmsecond import main as run_glmmsecond
from src.utils import skip_run

ROOT   = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "config_analysis.yml"

if __name__ == "__main__":
	with open(CONFIG) as f:
		cfg = yaml.safe_load(f)

	with skip_run("skip", "xdf") as check, check():
		run_xdf()

	with skip_run("run", "fixations") as check, check():
		run_fixations(cfg)

	with skip_run("skip", "saccades") as check, check():
		run_saccades(cfg)

	with skip_run("skip", "glmm") as check, check():
		run_from_config()

	with skip_run("skip", "glmmsecond") as check, check():
		run_glmmsecond()
