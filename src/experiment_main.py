from pathlib import Path

import ray
import yaml
from ixp.experiment import Experiment
from ixp.surveys.nasa_tlx import NasaTLX
from ixp.surveys.sart import SART
from utils import skip_run
from experiment.game import SARGame
from ixp.sensors.eye_tracker.tobii import TobiiEyeTracker
from ixp.individual_difference.mot import MOT
from ixp.individual_difference.vs import VS
from experiment.tutorial import SARTutorial

with Path("configs/experiment.yaml").open() as f:
    config = yaml.safe_load(f)


with skip_run('skip', 'tobii') as check, check():
    tobii= TobiiEyeTracker()
    tobii.initialize()
    tobii.calibrate()



with skip_run('run', 'tobii') as check, check():
    ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
    experiment = Experiment(config)

    experiment.add_task(
        name='visual_search',
        task_cls=VS,
        task_config={'config': config['vs']},
        order=1,
        instructions=[
            'Welcome to the experiment!\n\n',
            'In this session you will complete a visual search task. In this task, you will see a group of **T-shaped symbols** on the screen. Your goal is to find the **correct T** as quickly and accurately as possible. The distractors will also look like **T** shapes, but they will be **asymmetric**. You should choose the direction of that correct "T" by pressing the corresponding arrow key on your keyboard (up, down, left, or right). Please try to:  - focus carefully on the details of the T shapes,  respond as quickly and accurately as possible',
        ],
    )
    experiment.add_task(
        name='multi_object_tracking',
        task_cls=MOT,
        task_config={'config': config['mot']},
        order=2,
        instructions=[
            "In this task, several **blackcircles** will appear on the screen. At the beginning of each trial, a small number of these circles will be highlighted as **targets**. After that, all circles will begin moving. Your job is to **keep track of the target circles while they move**. At the end of the trial, the circles will stop, and you will be asked to identify the original target circles. Please try to: - focus carefully on the highlighted target circles, keep track of them as they move, and respond as accurately as possible",
        ],
    )
    experiment.add_task(
        name="practice",
        task_cls=SARTutorial,
        task_config={"config": config["game"]},
        order=2,
        instructions=["In this session you will complete a practice task to familiarize yourself with the interface and controls.",
        ],
    )
    experiment.add_task(
        name="main_game",
        task_cls=SARGame, 
        task_config={"config": config["game"]},
        order=3,
        instructions=[
            """rescue victims quickly and accurately while avoiding hazards.
Controls: turn with left/right (`<`/`>`), move forward, **Space** to interact with doors, **Shift** to drop key, **Tab** to pick up victim.
Rules: you cannot move backward; carry only one key at a time; closed doors (`_`) need a matching key; open doors are shown as `o`.
Victims: find the correct symmetric T (distractors are asymmetric T-shapes), move adjacent, then press **Tab**.
Goal: explore efficiently, open doors as needed, avoid fire tiles, rescue correct victims, and minimize unnecessary movement.
an AI teammate provides guidance in the chatbox. Above it you see victims rescued, score, carried key, steps used, and steps remaining.""",
        ],
    )
    experiment.add_task(
        name="sart",
        task_cls=SART,
        task_config={"config": config["surveys"]},
        order=5,
        instructions=[
            "SART\n\nYou will see a series of question please use the slider to answer.",
        ],
    )
    experiment.add_task(
        name="nasa_tlx",
        task_cls=NasaTLX,
        task_config={"config": config["surveys"]},
        order=6,
        instructions="NASA-TLX\n\nYou will rate your mental workload across several dimensions.",
    )

    # Run the experiment
    experiment.run()
    experiment.close()
