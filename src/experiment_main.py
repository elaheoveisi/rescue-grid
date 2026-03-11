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

    # experiment.add_task(
    #     name='visual_search',
    #     task_cls=VS,
    #     task_config={'config': config['vs']},
    #     order=1,
    #     instructions=[
    #         'Welcome to the experiment!\n\n',
    #         'In this session you will complete a visual search task.',
    #     ],
    # )
    # experiment.add_task(
    #     name='multi_object_tracking',
    #     task_cls=MOT,
    #     task_config={'config': config['mot']},
    #     order=2,
    #     instructions=[
    #         'In this task, several **red circles** will appear on the screen. At the beginning of each trial, a small number of these circles will be highlighted as **targets**. After that, all circles will begin moving. ',
    #     ],
    # )
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
            "In this session you will complete the main task",
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
