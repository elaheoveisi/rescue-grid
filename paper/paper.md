---
title: "MOSAIC: A Modular Search-and-Rescue Testbed for Human-AI Collaboration Research"
tags:
  - Python
  - human-AI collaboration
  - human-AI teaming
  - search and rescue
  - reinforcement learning
  - Gymnasium
  - MiniGrid
  - human-subjects experimentation
authors:
  - name: Bennett Dogbey
    orcid: 0000-0000-0000-0000 # TODO: real ORCID
    corresponding: true # TODO: confirm corresponding author
    affiliation: 1
  - name: Elahe Oveisi
    orcid: 0000-0000-0000-0000 # TODO: real ORCID
    affiliation: 1
  - name: Hemanth Manjunatha
    orcid: 0000-0003-2207-8960 # TODO: real ORCID
    affiliation: 1
affiliations:
  - name: Department of Mechanical and Aerospace Engineering, Oklahoma State University, Stillwater, OK
    index: 1

date: 1 September 2026 # TODO: update to the actual submission date
bibliography: paper.bib
---

# Summary

Research on human–AI collaboration requires controlled environments that are complex enough to produce meaningful decisions yet repeatable enough to support systematic experimentation. MOSAIC is an open-source, grid-based search-and-rescue platform designed for this purpose. Built on MiniGrid and Gymnasium [@chevalier2023minigrid; @towers2026gymnasium], it places a human participant in a time-constrained, multi-room mission involving real victims, decoys, locked doors, keys, and environmental hazards. During the mission, an AI assistant can provide natural-language recommendations based on the current task state, while the participant retains responsibility for deciding whether and how to follow that advice.

MOSAIC records structured information about the environment, participant actions, mission progress, and interactions with the AI assistant. Researchers can vary the environment layout, victim and hazard placement, visual perspective, reward structure, AI client, prompting method, and experimental pacing without rewriting the underlying task. The software separates reusable search-and-rescue mechanics from study-specific configuration and provides an experiment layer for integrating participant measures, synchronized sensing, and session replay. MOSAIC therefore supports reproducible investigations of how people seek, interpret, trust, and act on AI assistance under workload, uncertainty, and time pressure.

<!--Need to combine the above text with below-->

MOSAIC is a modular, grid-based Search and Rescue (SAR) simulation platform for studying
human-AI collaboration. Built on top of MiniGrid [@minigrid] and Gymnasium [@gymnasium], it
places a human or AI agent (or both, cooperating) in a multi-room building on fire, where real
victims (rendered as a cross) must be distinguished from decoy victims (rendered as a T-shape)
and rescued before a time limit expires, while navigating locked doors and lava hazards. The
platform separates a generic, installable core (`mosaic`) — environment dynamics, victim and
hazard placement, camera strategies, and GUI components — from study-specific orchestration
(`experiment`), which supplies calibrated pacing, tuned placement policies, and integration with
physiological sensing. Every major component (victim/hazard placers, the LLM client, and GUI
panels) is injected via constructor parameters rather than subclassed, so a new study can swap
in custom behavior without forking the core package.

# Statement of need

Research on human-AI interaction requires tasks that simulate realistic challenges such as time pressure, incomplete information, and tangible consequences for errors, while maintaining enough control for consistent replication across participants. Simply measuring the final task score is insufficient; researchers need detailed data on the sequence of participant actions, the timing and content of AI advice, and aligned physiological signals.

Currently, assembling this experimental setup is a labor-intensive process for each research. It involves integrating multiple components such as simulators, participant interfaces, AI modules, sensor synchronization, logging, and replay tools that were often developed independently without a shared timing framework. This leads to substantial, often redundant framework that is tightly coupled to specific study designs, limiting reusability.

MOSAIC provides an integrated, Gymnasium-compatible framework for human-AI teaming, human factors, and reinforcement learning research. It features a configurable search-and-rescue task, operator interface, and modular advisory channel, with consistent logging and replay tied to environment observations. Task parameters such as reward scales, pacing, hazard density, and sensor settings are injected at runtime to maintain neutrality and adaptability.

By separating core task mechanics from experiment layers and supporting replaceable components, MOSAIC enables researchers to manipulate conditions while preserving a consistent task foundation, facilitating rigorous study of AI-assisted decision-making under realistic constraints.

<!--Need to combine the above text with below-->

Studying how humans and AI systems collaborate under time pressure and uncertainty requires an
environment that is simultaneously (1) a valid reinforcement-learning testbed, (2) instrumented
enough for cognitive/behavioral research, and (3) flexible enough to support an LLM-based
assistant without hard-coding a single provider. Existing grid-world environments such as
MiniGrid provide the RL substrate but no built-in support for eye-tracking synchronization,
LLM-driven advisory agents, or the mixed-initiative "rescue" task structure needed for human-AI
teaming studies; general-purpose HCI experiment frameworks, conversely, do not provide a
Gymnasium-compatible environment. MOSAIC addresses this gap by combining a Gymnasium-compatible
SAR environment with Lab Streaming Layer (LSL) integration for eye-tracking, a
provider-agnostic LLM client contract for advisory agents, and standard human-factors measures
(NASA-TLX [@hart1988nasa]; SART [@taylor1990situational]) already wired into its experiment
framework. Researchers in human-AI teaming, situation awareness, and mixed-initiative RL can use
MOSAIC's neutral core directly, or fork the pattern already used by `experiment/` to calibrate
their own study without modifying the installable package.


# State of the field

Simulation testbeds provide controlled and flexible environments for studying human–AI interaction, allowing researchers to systematically manipulate task conditions and reproduce experimental scenarios. They also enable researchers to modify task parameters, introduce new features, and adapt the environment to the requirements of different experimental designs.


# Software design

<!--
OUTLINE - 300-400 words. Do not draft yet.

Organize the discussion around design decisions, not a module tour:

   1. Reusable Gymnasium/MiniGrid environment core.
   2. SAR mechanics and observations.
   3. Separation between reusable `mosaic` code and study-specific `experiment`
      code.
   4. Constructor-injected study components and neutral defaults.
   5. Human interaction through the GUI.
   6. Provider-independent AI-advisory interface.
   7. Experimental logging, sensing, and replay.
   8. Composition versus inheritance.
   9. Reusability versus study-specific calibration.
  10. Reproducibility and testability.

Grounding pointers for the drafter - verify each against the code before making
any claim about it:
  - src/mosaic/sar/env.py, src/mosaic/sar/observations.py, src/mosaic/sar/actions.py
  - src/mosaic/core/level.py, src/mosaic/core/camera.py, src/mosaic/core/placers.py
  - src/mosaic/gui/main.py (SAREnvGUI), src/mosaic/gui/chat.py,
    src/mosaic/gui/feedback.py
  - src/mosaic/llm/client.py, src/mosaic/llm/parser.py
  - src/experiment/experiment.py, src/experiment/replay.py
  - src/experiment/sensors/eye_tracker/
  - tests/
-->

<!--
RESERVED - one optional architecture figure. NOT CREATED YET.

Intended content:

    Human / AI
        |
    GUI and advisory interface
        |
    MOSAIC SAR environment
        |
    Observations, events and measurements
        |
    Logging, sensing and replay

When the figure exists, place it in paper/figures/ and reference it as:
    ![Caption.\label{fig:architecture}](figures/architecture.png)
-->

# Research impact statement


MOSAIC lowers the engineering barrier to conducting controlled human–LLM (Large Language Model) teaming research by offering an integrated, reusable experimental platform. At its core is a dynamic, multi-room search-and-rescue task where human operators navigate environments, unlock doors, avoid hazards, and distinguish real victims from fake under time pressure and uncertainty. Built to standard reinforcement learning specifications via a Gymnasium-compatible interface, MOSAIC allows AI advisory agents to interact seamlessly within the simulation environment alongside human subjects.   
Medium


Rather than requiring researchers to build game mechanics, AI integration, and telemetry from scratch, MOSAIC provides a unified system with modular architecture. Key parameters, including task layouts, hazard and victim placement, difficulty pacing, reward structures, visual perspectives, choice of AI provider, and prompting strategies, can be dynamically configured without modifying core task code. Simultaneously, the platform logs detailed behavioral data, enables full trial replays, and synchronizes physiological sensors, such as eye-tracking and EEG devices.   


MOSAIC's versatility has already supported empirical human–AI collaboration studies. In work accepted for IEEE SMC 2026 and available in Arxiv[@oveisi2026llm]  , researchers used the platform to evaluate how user expertise affects visual attention and task planning under LLM guidance.Other studies [@oveisillm] leveraged MOSAIC’s physiological monitoring capabilities to analyze real-time cognitive responses during task execution.

Together, these applications show MOSAIC’s potential for multimodal investigations into how humans evaluate, verify, and act upon AI guidance. Future applications include controlled studies on trust calibration, AI reliability, workload management, situation awareness, error detection, adaptive AI assistance, and reinforcement learning. Designed as an open, modular framework, MOSAIC promotes reproducible, cross-disciplinary research across human factors, human-computer interaction (HCI), and neuroergonomics.



<!--
OUTLINE - 150-250 words. Do not draft yet.

AWAITING AUTHOR CONFIRMATION. Request verified evidence only:
  - Completed or ongoing studies that used MOSAIC.
  - Pilot-study use and approved participant information.
  - Data streams actually collected and the research questions they enabled.
  - Publications, preprints, posters, presentations, theses, or datasets.
  - Use by collaborators or other research groups.
  - Reproducible demonstrations or benchmarks.

CONSTRAINT: projected or planned future applications are NOT evidence of impact
and must not be presented as such. Do not invent studies, participant counts,
or results.
-->

# AI usage disclosure

<!--
OUTLINE - 60-100 words. Do not draft yet.

AWAITING AUTHOR CONFIRMATION of:
  - Which generative-AI tools were used in the software, the documentation, or
    the manuscript.
  - What those tools contributed.
  - What the human authors reviewed and decided.
  - How tests, code review, and manual verification checked correctness.

Do not write the final disclosure until the authors confirm these facts.
-->

# Acknowledgements

<!--
OUTLINE - 40-80 words. Do not draft yet.

AWAITING AUTHOR CONFIRMATION of:
  - Funding agencies and grant numbers.
  - Institutional and laboratory support.
  - Non-author contributors.
  - Equipment support.

Do not invent acknowledgements, funders, or grant numbers.
-->

# References

