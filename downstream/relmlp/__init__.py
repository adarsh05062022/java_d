"""DeepSets downstream probe: a relational classifier with no hand-built aggregates.

This subpackage is deliberately isolated from downstream/{evaluate,run,tasks}.py. It reuses
their data loaders and metric functions so the numbers stay comparable, but the XGBoost path is
never touched. See downstream/relmlp/README.md.
"""
