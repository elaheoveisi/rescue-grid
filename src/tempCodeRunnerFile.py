with skip_run("skip", "sar_experiment_test") as check, check():
#     ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
#     experiment = Experiment(config)

#     experiment.add_task(
#         name="main_game",
#         task_cls=SARGame,
#         task_config={"config": config["game"]},
#         order=3,
#         instructions=instructions["main_game"],
#     )

#     # Run the experiment
#     experiment.run()
#     experiment.close()
