"""
Single source of truth for experiment-wide constants.

Every script (training, calibration eval, OOD eval, ablations) should import
OOD_CLASSES from here rather than hardcoding the list. If train and eval scripts
disagree on which classes were held out, the OOD experiment is silently invalid.
"""

# Classes held out entirely from train/val. The model never sees these during training.
# Chosen to be realistic robotic-grasping targets (household/office objects) with
# diverse geometry (round/hollow, flat/rigid, irregular) and reasonable test-set sizes.
OOD_CLASSES = ["bottle", "bowl", "cup", "keyboard", "laptop"]

# Reproducibility
SEED = 42

# Dataset
NUM_POINTS = 1024
VAL_FRACTION = 0.1
