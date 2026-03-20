from analysis.glmm import run_from_config
from src.utils import skip_run

if __name__ == "__main__":
	with skip_run("run", "glmm") as check, check():
		run_from_config()
