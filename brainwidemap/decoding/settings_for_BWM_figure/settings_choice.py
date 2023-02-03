import numpy as np
from pathlib import Path

from brainwidemap.decoding.functions.process_targets import optimal_Bayesian
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
from sklearn import linear_model as lm

"""
------------------------------------------------
ADAPT AT LEAST THESE IN YOUR COPY OF SETTINGS.PY
------------------------------------------------
"""
RESULTS_DIR = Path("/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map")
# Directory to which to save all results and outputs, including models. Will be created if it doesn't exist.
SLURM_DIR = Path("/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm")
# Directory where slurm output and error files will be saved

DATE = '18-01-2023'
# Either current date for a fresh run, or date of the run you want to build on
# Date must be different if you do different runs of the same target
# e.g. signcont side with LogisticRegression vs signcont with Lasso

TARGET = 'choice'
# single-bin targets:
#   'pLeft' - estimate of block prior
#   'signcont' - signed contrast of stimulus
#   'choice' - subject's choice (L/R)
#   'feedback' - correct/incorrect
# multi-bin targets:
#   'wheel-vel' - wheel velocity
#   'wheel-speed' - wheel speed
#   'l-whisker-me' - motion energy of left whisker pad
#   'r-whisker-me' - motion energy of right whisker pad
"""
------------------------------------------------
"""

MODEL = expSmoothing_prevAction
# behavioral model used for pLeft
# - expSmoothing_prevAction (not string)
# - expSmoothing_stimside (not string)
# - optBay  (not string)
# - oracle (experimenter-defined 0.2/0.8)
# - absolute path; this will be the interindividual results
BEH_MOUSELEVEL_TRAINING = False  # If True, trains the behavioral model session-wise else mouse-wise

if TARGET == 'pLeft':
    ALIGN_TIME = 'stimOn_times'  # event on which windows are aligned
    TIME_WINDOW = (-0.4, -0.1)  # start and end of decoding window, relative to ALIGN_TIME (seconds)
    BINSIZE = 0.3  # size of individual bins within time window (seconds)
    N_BINS_LAG = None  # number of bins to use for prediction
    USE_IMPOSTER_SESSION = False  # False will use pseudo-sessions to create null distributions
    BINARIZATION_VALUE = 0.5  # to binarize the target -> could be useful with logistic regression estimator
    TANH_TRANSFORM = False
    EXCLUDE_UNBIASED_TRIALS = True  # True to take out unbiased block at beginning of session
elif TARGET == 'signcont':
    ALIGN_TIME = 'stimOn_times'
    TIME_WINDOW = (0.0, 0.1)
    BINSIZE = 0.1
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    BINARIZATION_VALUE = None
    TANH_TRANSFORM = True
    EXCLUDE_UNBIASED_TRIALS = False
elif TARGET == 'choice':
    ALIGN_TIME = 'firstMovement_times'
    TIME_WINDOW = (-0.1, 0.0)
    BINSIZE = 0.1
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    BINARIZATION_VALUE = 0 # choice vals are -1 and 1
    TANH_TRANSFORM = False
    EXCLUDE_UNBIASED_TRIALS = False
elif TARGET == 'feedback':
    ALIGN_TIME = 'feedback_times'
    TIME_WINDOW = (0.0, 0.2)
    BINSIZE = 0.2
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    BINARIZATION_VALUE = 0 # feedback vals are -1 and 1
    TANH_TRANSFORM = False
    EXCLUDE_UNBIASED_TRIALS = False
elif TARGET in ['wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me']:
    ALIGN_TIME = 'firstMovement_times'
    TIME_WINDOW = (-0.2, 1.0)
    BINSIZE = 0.02
    N_BINS_LAG = 10
    USE_IMPOSTER_SESSION = True
    BINARIZATION_VALUE = None
    TANH_TRANSFORM = False
    EXCLUDE_UNBIASED_TRIALS = False


# DECODER PARAMS
ESTIMATOR = lm.LogisticRegression
# A scikit learn linear_model class: LinearRegression, Lasso (linear + L1), Ridge (linear + L2) or LogisticRegression
ESTIMATOR_KWARGS = {'tol': 0.001, 'max_iter': 1000, 'fit_intercept': True}  # default args for decoder
if ESTIMATOR == lm.LogisticRegression:
    ESTIMATOR_KWARGS = {**ESTIMATOR_KWARGS, 'penalty': 'l1', 'solver': 'liblinear'}
    HPARAM_GRID = {'C': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}  # hyperparameter values to search over
else:
    HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
N_PSEUDO = 200  # number of pseudo/imposter sessions to fit per session
N_PSEUDO_PER_JOB = 100  # number of pseudo/imposter sessions to assign per cluster job
N_JOBS_PER_SESSION = N_PSEUDO // N_PSEUDO_PER_JOB  # number of cluster jobs to run per session
N_RUNS = 10  # number of times to repeat full nested xv with different folds
SHUFFLE = True  # true for interleaved xv, false for contiguous
QUASI_RANDOM = False  # if True, decoding is launched in a quasi-random, reproducible way => it sets the seed
BALANCED_WEIGHT = True  # seems to work better with BALANCED_WEIGHT=False, but putting True is important
BALANCED_CONTINUOUS_TARGET = False  # is target continuous or discrete FOR BALANCED WEIGHTING

# CLUSTER/UNIT PARAMS
MIN_UNITS = 10  # regions with units below this threshold are skipped
SINGLE_REGION = True  # perform decoding on region-wise or whole-brain decoding
MERGED_PROBES = True  # merge probes before performing analysis

# SESSION/BEHAVIOR PARAMS
MIN_BEHAV_TRIAS = 200  # minimum number of behavioral trials completed in one session, that fulfill below criteria
MIN_RT = 0.08  # remove trials with reaction times above/below these values (seconds), if None, don't apply
MAX_RT = 2.0
MIN_LEN = None  # remove trials with length (feedback_time-goCue_time) above/below these value, if None, don't apply
MAX_LEN = None

# NULL DISTRIBUTION PARAMS
CONSTRAIN_NULL_SESSION_WITH_BEH = False  # TODO
STITCHING_FOR_IMPOSTER_SESSION = True  # If true, stitches sessions to create imposters
USE_IMPOSTER_SESSION_FOR_BALANCING = False  # Not currently implemented, so it will be forced to be False
MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION = 700
# Constrain the number of trials per session to have more potential imposter sessions to use in the null distribution

# MISC PARAMS
USE_OPENTURNS = False  # uses openturns to perform kernel density estimation
BIN_SIZE_KDE = 0.05
SAVE_PREDICTIONS = True  # save model predictions in output file
SAVE_PREDICTIONS_PSEUDO = False  # save model predictions in output file from pseudo/imposter/synthetic sessions
SAVE_BINNED = True  # save binned neural predictors in output file for non-null fits (causes large files)
EXCLUDE_TRIALS_WITHIN_VALUES = (None, None) # Applies mask equally to target and control, only works for scalars
MIN_SESS_PER_REG = 2

"""
----------
RUN CHECKS
-----------
"""

target_options = ['pLeft', 'choice', 'feedback', 'signcont', 'wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me']
if TARGET not in target_options:
    raise NotImplementedError(f"Provided target option '{TARGET}' invalid; must be in {target_options}")

if BINARIZATION_VALUE and TANH_TRANSFORM:
   raise ValueError("Binarization can be done without tanh_transform; do not choose both")

modeldispatcher = {
    expSmoothing_prevAction: expSmoothing_prevAction.name,
    expSmoothing_stimside: expSmoothing_stimside.name,
    optimal_Bayesian: 'optBay',
    None: 'oracle'
}
if MODEL not in list(modeldispatcher.keys()) and not isinstance(MODEL, str):
    raise NotImplementedError(f"Provided model '{MODEL}' is not supported yet.")

estimator_options = [lm.LinearRegression, lm.Lasso, lm.Ridge, lm.LogisticRegression]
estimator_strs = ['LinearRegression', 'Lasso', 'Ridge', 'LogisticsRegression']
if ESTIMATOR not in estimator_options:
    raise NotImplementedError(f"Provided estimator '{ESTIMATOR}' invalid; must be in {estimator_options}")

align_event_options = ['firstMovement_times', 'goCue_times', 'stimOn_times', 'feedback_times']
if ALIGN_TIME not in align_event_options:
    raise NotImplementedError(f"Provided align event '{ALIGN_TIME}' invalid; must be in {align_event_options}")

# ValueErrors and NotImplementedErrors
if TARGET in ['choice', 'feedback'] and (MODEL != expSmoothing_prevAction or USE_IMPOSTER_SESSION):
    raise ValueError('To decode choice or feedback, you must use the actionKernel model and frankenstein sessions')

if ESTIMATOR == lm.LogisticRegression and BALANCED_CONTINUOUS_TARGET:
    raise ValueError('You can not have a continuous target with logistic regression')

if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('Full probes analysis can only be done with merged probes')

ADD_TO_SAVING_PATH = (
        f'imposterSess_{int(USE_IMPOSTER_SESSION)}_balancedWeight_{int(BALANCED_WEIGHT)}_'
        f'RegionLevel_{int(SINGLE_REGION)}_mergedProbes_{int(MERGED_PROBES)}_behMouseLevelTraining_'
        f'{int(BEH_MOUSELEVEL_TRAINING)}_constrainNullSess_{int(CONSTRAIN_NULL_SESSION_WITH_BEH)}'
)

"""
---------------------
CONSTRUCT PARAMS DICT
---------------------
"""

params = {
    'date': DATE,
    # TARGET
    'target': TARGET,
    'model': MODEL,
    'modeldispatcher': modeldispatcher,
    'beh_mouseLevel_training': BEH_MOUSELEVEL_TRAINING,
    'binarization_value': BINARIZATION_VALUE,
    'tanh_transform': TANH_TRANSFORM,
    # TIME WINDOW
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'binsize': BINSIZE,
    'n_bins_lag': N_BINS_LAG,
    # DECODER
    'estimator': ESTIMATOR,
    'estimator_kwargs': ESTIMATOR_KWARGS,
    'hyperparam_grid': HPARAM_GRID,
    'n_pseudo': N_PSEUDO,
    'n_pseudo_per_job': N_PSEUDO_PER_JOB,
    'n_runs': N_RUNS,
    'shuffle': SHUFFLE,
    'quasi_random': QUASI_RANDOM,
    'balanced_weight': BALANCED_WEIGHT,
    'balanced_continuous_target': BALANCED_CONTINUOUS_TARGET,
    # CLUSTER/UNIT
    'min_units': MIN_UNITS,
    'single_region': SINGLE_REGION,
    'merged_probes': MERGED_PROBES,
    # SESSION/BEHAVIOR
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'min_rt': MIN_RT,
    'max_rt': MAX_RT,
    'min_len': MIN_LEN,
    'max_len': MAX_LEN,
    'exclude_unbiased_trials': EXCLUDE_UNBIASED_TRIALS,
    'exclude_trials_within_values': EXCLUDE_TRIALS_WITHIN_VALUES,
    # NULL DISTRIBUTION
    'use_imposter_session': USE_IMPOSTER_SESSION,
    'constrain_null_session_with_beh': CONSTRAIN_NULL_SESSION_WITH_BEH,
    'stitching_for_imposter_session': STITCHING_FOR_IMPOSTER_SESSION,
    'use_imposter_session_for_balancing': False,
    'max_number_trials_when_no_stitching_for_imposter_session':
        MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION,
    # MISC
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE,
    'save_predictions': SAVE_PREDICTIONS,
    'save_predictions_pseudo': SAVE_PREDICTIONS_PSEUDO,
    'save_binned': SAVE_BINNED,
    'add_to_saving_path': ADD_TO_SAVING_PATH,
    'imposterdf': None,
    'min_sess_per_reg': MIN_SESS_PER_REG,
}

"""
------------------------------
CONSTRUCT SETTINGS FORMAT NAME
------------------------------
"""
estimatorstr = [estimator_strs[i] for i in range(len(estimator_options)) if ESTIMATOR == estimator_options[i]]
assert len(estimatorstr)==1
estimatorstr = estimatorstr[0]
start_tw, end_tw = TIME_WINDOW

model_str = 'interIndividual' if isinstance(MODEL, str) else modeldispatcher[MODEL]


SETTINGS_FORMAT_NAME = str(RESULTS_DIR.joinpath('decoding','results','neural','ephys',
                              '_'.join([DATE, 'decode', TARGET,
                               model_str if TARGET in ['prior','pLeft'] else 'task',
                               estimatorstr,
                               'align', ALIGN_TIME,
                               str(N_PSEUDO), 'pseudosessions',
                               'regionWise' if SINGLE_REGION else 'allProbes',
                               'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_')])))
if ADD_TO_SAVING_PATH != '':
    SETTINGS_FORMAT_NAME = SETTINGS_FORMAT_NAME + '_' + ADD_TO_SAVING_PATH


