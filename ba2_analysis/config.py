from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
REAL_ROOT = DATASET_ROOT / "real-world"
# Simulator dropped from BA2: its GPS is largely static (stationary-bike setup),
# stress labels are not part of the BA2 scope, and clustering on simulator data
# collapses to IMU-only - which does not validate the real-world pipeline.

RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Subsampling. Set to None to use all trips.
MAX_TRIPS_PER_USER = None
MAX_USERS = None

# Preprocessing.
TARGET_HZ = 50.0
LOWPASS_CUTOFF_HZ = 10.0
LOWPASS_ORDER = 4

# Windowing.
WINDOW_SECONDS = 10.0
WINDOW_OVERLAP = 0.5

# FFT bands (Hz).
CADENCE_BAND = (1.0, 2.0)
VIBRATION_BAND = (5.0, 20.0)

# Clustering.
K_RANGE = list(range(2, 9))
RANDOM_STATE = 42

# Robustness / augmentation.
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]
DROPOUT_RATES = [0.0, 0.05, 0.1, 0.2]
DRIFT_LEVELS = [0.0, 0.01, 0.05, 0.1]
N_BOOTSTRAP = 20
BOOTSTRAP_FRACTION = 0.8

# Synthetic personas (for external metrics + AL ground truth).
SYNTH_N_USERS_PER_PERSONA = 4
SYNTH_TRIPS_PER_USER = 2
SYNTH_TRIP_MINUTES = 5.0

# Active learning (supervised classifier baseline; entropy sampling).
AL_INITIAL_LABELED = 30
AL_QUERIES_PER_ROUND = 20
AL_TOTAL_ROUNDS = 15

# GPS sanity filter: drop trips whose median speed exceeds this threshold
# (sustained GPS-jump trips are sensor artefacts, not cycling behaviour).
MAX_TRIP_MEDIAN_SPEED_MPS = 15.0  # 54 km/h

# Logging.
VERBOSE = True
