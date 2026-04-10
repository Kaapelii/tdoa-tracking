## simulation code implementation for the report.

| File | Description |
| --- | --- |
| `config.py` | Configuration params for the simulation. |
| `scenario.py` | Builds the motion and TDOA measurement model and simulates a single run. The truth path also has a small built-in maneuver so it curves a bit instead of looking almost straight. |
| `estimators.py` | The EKF, UKF and the two smoothers implementations. |
| `plotting.py` | Makes the figures that go into the report. |
| `run_experiment.py` | Runs one example and the Monte Carlo comparison, saves the plots, and prints the main results to the console. |
