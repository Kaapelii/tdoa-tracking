## simulation code implementation for the report.

`config.py` has the configuration params for the simulation.
`scenario.py` builds the motion and TDOA measurement model and simulates a single run. The truth path also has a small built-in maneuver so it curves a bit instead of looking almost straight.
`estimators.py` has the EKF, UKF and the two smoothers implementations.
`plotting.py` makes the figures that go into the report.
`run_experiment.py` runs one example and the Monte Carlo comparison, saves the plots, and prints the main results to the console.
