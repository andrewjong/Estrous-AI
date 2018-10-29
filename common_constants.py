EXPERIMENTS_ROOT = "experiments"
META_FNAME = "meta.json"
MODEL_PARAMS_FNAME = "model.pth"
TRAIN_RESULTS_FNAME = "train.csv"
PREDICT_RESULTS_FNAME = "predictions.csv"

# orders the phases according to the Estrous cycle by first letter
# (*p*roestrus, *e*strus, etc)
PHASE_ORDER = {'p': 1, 'e': 2, 'm': 3, 'd': 4}  # ordering of estrous cycle

MODEL_TO_IMAGE_SIZE = {
    "inceptionv4": 299,
    "inceptionresnetv2": 299,
    "nasnetalarge": 331,
    "pnasnet5large": 331,
    "polynet": 331,
}
DEFAULT_IMAGE_SIZE = 224
DEFAULT_VAL_PROPORTION = 0.15
DEFAULT_BATCH_SIZE = 16
