import logging
import numpy as np
import os
from brainwidemap.decoding.functions.process_targets import optimal_Bayesian
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
import sklearn.linear_model as sklm
import warnings

logger = logging.getLogger('ibllib')
logger.disabled = True


# DOCS: DECODING TARGET
# --------------------------------------------------
# TARGET: str
# decoding target
# single-bin targets: pLeft | signcont | choice | feedback
# multi-bin targets: wheel-vel | wheel-speed | pupil | l-paw-pos | l-paw-vel | l-paw-speed |
#    l-whisker-me | r-paw-pos | r-paw-vel | r-paw-speed | r-whisker-me

# MODEL: mixed
# behavioral model used for pLeft
# - expSmoothing_prevAction (not string)
# - expSmoothing_stimside (not string)
# - optBay  (not string)
# - oracle (experimenter-defined 0.2/0.8)
# - absolute path; this will be the interindividual results

# BEH_MOUSELEVEL_TRAINING: bool
# if true, trains the behavioral model session-wise else mouse-wise


# DOCS: TIME WINDOW PARAMS
# --------------------------------------------------
# ALIGN_TIME: str
# event on which windows are aligned
# - firstMovement_times
# - goCue_times
# - stimOn_times
# - feedback_times

# TIME_WINDOW: tuple
# start and end of decoding window, relative to ALIGN_TIME (seconds)

# BINSIZE: float
# size of individual bins within time window (seconds)

# N_BINS_LAG: int
# number of bins to use for prediction


# DOCS: DECODER PARAMS
# --------------------------------------------------
# ESTIMATOR: str
# - linear
# - lasso (linear + L1)
# - ridge (linear + L2)
# - logistic

# ESTIMATOR_KWARGS: dict
# default args for decoder

# HPARAM_GRID: dict
# hyperparameter values to search over

# N_PSEUDO: int
# number of pseudo/imposter sessions to fit per session

# N_PSEUDO_PER_JOB: int
# number of pseudo/imposter sessions to assign per cluster job

# N_RUNS: int
# number of times to repeat full nested xv with different folds

# SHUFFLE: bool
# true for interleaved xv, false for contiguous

# QUASI_RANDOM: bool
# if TRUE, decoding is launched in a quasi-random, reproducible way => it sets the seed

# BALANCED_WEIGHT: bool
# seems to work better with BALANCED_WEIGHT=False, but putting True is important

# BALANCED_CONTINUOUS_TARGET: bool
# is target continuous or discrete FOR BALANCED WEIGHTING


# DOCS: CLUSTER/UNIT PARAMS
# --------------------------------------------------
# MIN_UNITS: int
# regions with units below this threshold are skipped

# QC_CRITERIA: float
# fraction of qc criteria each unit needs to pass for inclusion

# SINGLE_REGION: bool
# perform decoding on region-wise or whole-brain decoding

# MERGED_PROBES: bool
# merge probes before performing analysis


# DOCS: SESSION/BEHAVIOR PARAMS
# --------------------------------------------------
# MIN_BEHAV_TRIAS: int
# minimum number of behavioral trials completed by subject

# MIN_RT/MAX_RT: float
# remove trials with reaction times below/above these values (seconds)

# MIN_LEN/MAX_LEN: float
# remove trials with durations below/above these values (seconds)

# NO_UNBIAS: bool
# true to take out unbiased block at beginning of session

# N_TRIALS_TAKEOUT_END: int
# number of trials to remove from the end of the session


# DOCS: NULL DISTRIBUTION PARAMS
# --------------------------------------------------
# USE_IMPOSTER_SESSION: bool
# false will use pseudo-sessions to create null distributions

# IMPOSTER_GENERATE_FROM_EPHYS: bool
# true to just use ephys sessions, false to use training sessions (more templates)

# CONSTRAIN_NULL_SESSION_WITH_BEH: bool
# TODO

# STITCHING_FOR_IMPOSTER_SESSION: bool
# if true, stitches sessions to create imposters

# USE_IMPOSTER_SESSION_FOR_BALANCING: bool
# if false, simulate the model (should be false)

# IMPOSTER_GENERATE_FAKE: bool
# testing purposes?

# FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION: bool
# TODO

# MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION: int
# this is a constraint on the number of trials of a session to insure that there will more
# potential imposter sessions to use in the null distribution


# DOCS: MISC PARAMS
# --------------------------------------------------
# USE_OPENTURNS: bool
# BIN_SIZE_KDE: float
# uses openturns to perform kernel density estimation

# SAVE_PREDICTIONS: bool
# save model predictions in output file

# SAVE_PREDICTIONS_PSEUDO: bool
# save model predictions in output file from pseudo/imposter/synthetic sessions

# SAVE_BINNED: bool
# save binned neural predictors in output file (causes large files)

# BINARIZATION_VALUE
# to binarize the target -> could be useful with logistic regression estimator

# MOTOR_REGRESSORS: bool
# add DLC data as additional regressors to neural activity

# MOTOR_REGRESSORS_ONLY: bool
# *only* use motor regressors, no neural activity


# SELECT PARAMS
# --------------------------------------------------
DATE = '08-10-2022'
TARGET = 'wheel-speed'

if TARGET == 'pLeft':
    MODEL = None
    ALIGN_TIME = 'stimOn_times'
    TIME_WINDOW = (-0.4, -0.1)
    BINSIZE = 0.3
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
elif TARGET == 'signcont':
    MODEL = None
    ALIGN_TIME = 'stimOn_times'
    TIME_WINDOW = (0.0, 0.1)
    BINSIZE = 0.1
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    raise NotImplementedError
elif TARGET == 'choice':
    MODEL = None
    ALIGN_TIME = 'firstMovement_times'
    TIME_WINDOW = (-0.1, 0.0)
    BINSIZE = 0.1
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    raise NotImplementedError
elif TARGET == 'feedback':
    MODEL = None
    ALIGN_TIME = 'feedback_times'
    TIME_WINDOW = (0.0, 0.2)
    BINSIZE = 0.2
    N_BINS_LAG = None
    USE_IMPOSTER_SESSION = False
    raise NotImplementedError
elif TARGET in ['wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me']:
    MODEL = None
    ALIGN_TIME = 'firstMovement_times'
    TIME_WINDOW = (-0.2, 1.0)
    BINSIZE = 0.02
    N_BINS_LAG = 10
    USE_IMPOSTER_SESSION = True
else:
    raise NotImplementedError


BEH_MOUSELEVEL_TRAINING = False

# DECODER PARAMS
ESTIMATOR = 'lasso'
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 20000, 'fit_intercept': True}
if ESTIMATOR == 'logistic':
    HPARAM_GRID = {'C': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
else:
    HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
N_PSEUDO = 1
N_PSEUDO_PER_JOB = 1
N_RUNS = 5
SHUFFLE = True
QUASI_RANDOM = True
BALANCED_WEIGHT = False
BALANCED_CONTINUOUS_TARGET = True

# CLUSTER/UNIT PARAMS
MIN_UNITS = 10
QC_CRITERIA = 3 / 3
SINGLE_REGION = True
MERGED_PROBES = False

# SESSION/BEHAVIOR PARAMS
MIN_BEHAV_TRIAS = 200
MIN_RT = 0.08  # 0.08  # Float (s) or None
MAX_RT = None
MIN_LEN = None
MAX_LEN = None
NO_UNBIAS = False
N_TRIALS_TAKEOUT_END = 0

# NULL DISTRIBUTION PARAMS
IMPOSTER_GENERATE_FROM_EPHYS = False
CONSTRAIN_NULL_SESSION_WITH_BEH = False
STITCHING_FOR_IMPOSTER_SESSION = True
USE_IMPOSTER_SESSION_FOR_BALANCING = False
IMPOSTER_GENERATE_FAKE = False
FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION = False
MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION = 700

# MISC PARAMS
USE_OPENTURNS = False
BIN_SIZE_KDE = 0.05
SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_PSEUDO = False
SAVE_BINNED = False
BINARIZATION_VALUE = None
MOTOR_REGRESSORS = False
MOTOR_REGRESSORS_ONLY = False  # only use motor regressors


# RUN CHECKS
# --------------------------------------------------

# decoding targets
target_options_singlebin = [
    'pLeft',  # some estimate of the block prior
    'choice',  # subject's choice (L/R)
    'feedback',  # correct/incorrect
    'signcont',  # signed contrast of stimulus
]
target_options_multibin = [
    'wheel-vel',
    'wheel-speed',
    'pupil',
    # 'l-paw-pos',
    # 'l-paw-vel',
    # 'l-paw-speed',
    'l-whisker-me',
    # 'r-paw-pos',
    # 'r-paw-vel',
    # 'r-paw-speed',
    'r-whisker-me',
]

# options for behavioral models
behavior_model_options = [
    'expSmoothing_prevAction',
    'optimal_Bayesian',
    'oracle',
    '<absolute_filepath>',
]

# options for align events
align_event_options = [
    'firstMovement_times',
    'goCue_times',
    'stimOn_times',
    'feedback_times',
]

# options for decoder
decoder_options = {
    'linear': sklm.LinearRegression,
    'lasso': sklm.Lasso,
    'ridge': sklm.Ridge,
    'logistic': sklm.LogisticRegression
}

N_JOBS_PER_SESSION = N_PSEUDO // N_PSEUDO_PER_JOB

if TARGET not in target_options_singlebin + target_options_multibin:
    raise NotImplementedError('provided target option \'{}\' invalid; must be in {}'.format(
        TARGET, target_options_singlebin + target_options_multibin,
    ))

modeldispatcher = {
    expSmoothing_prevAction: expSmoothing_prevAction.name,
    expSmoothing_stimside: expSmoothing_stimside.name,
    optimal_Bayesian: 'optBay',
    None: 'oracle'
}

if MODEL not in list(modeldispatcher.keys()) and not isinstance(MODEL, str):
    raise NotImplementedError('this MODEL is not supported yet')

if ALIGN_TIME not in align_event_options:
    raise NotImplementedError('provided align event \'{}\' invalid; must be in {}'.format(
        ALIGN_TIME, align_event_options,
    ))

if ESTIMATOR not in decoder_options.keys():
    raise NotImplementedError('provided estimator \'{}\' invalid; must be in {}'.format(
        ESTIMATOR, decoder_options,
    ))
else:
    ESTIMATOR = decoder_options[ESTIMATOR]

# ValueErrors and NotImplementedErrors
if TARGET in ['choice', 'feedback'] and (MODEL != expSmoothing_prevAction or USE_IMPOSTER_SESSION):
    raise ValueError(
        f'if you want to decode choice or feedback, you must use the actionKernel model and '
        f'frankenstein sessions')

if USE_IMPOSTER_SESSION_FOR_BALANCING:
    raise ValueError(
        f'this is not implemented yet -- or it is but unsure of the state given recent code '
        f'featuring')

if ESTIMATOR == sklm.LogisticRegression and BALANCED_CONTINUOUS_TARGET:
    raise ValueError('you can not have a continuous target with logistic regression')

if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('full probes analysis can only be done with merged probes')


ADD_TO_SAVING_PATH = (
    'imposterSess_%i_balancedWeight_%i_RegionLevel_%i_mergedProbes_%i_behMouseLevelTraining_%i_constrainNullSess_%i'
    % (USE_IMPOSTER_SESSION, BALANCED_WEIGHT, SINGLE_REGION, MERGED_PROBES,
       BEH_MOUSELEVEL_TRAINING, CONSTRAIN_NULL_SESSION_WITH_BEH))


kwargs = {
    'date': DATE,
    # TARGET
    'target': TARGET,
    'model': MODEL,
    'modeldispatcher': modeldispatcher,
    'beh_mouseLevel_training': BEH_MOUSELEVEL_TRAINING,
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
    'qc_criteria': QC_CRITERIA,
    'single_region': SINGLE_REGION,
    'merged_probes': MERGED_PROBES,
    # SESSION/BEHAVIOR
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'min_rt': MIN_RT,
    'max_rt': MAX_RT,
    'min_len': MIN_LEN,
    'max_len': MAX_LEN,
    'no_unbias': NO_UNBIAS,
    'n_trials_takeout_end': N_TRIALS_TAKEOUT_END,
    # NULL DISTRIBUTION
    'use_imposter_session': USE_IMPOSTER_SESSION,
    'imposter_generate_from_ephys': IMPOSTER_GENERATE_FROM_EPHYS,
    'constrain_null_session_with_beh': CONSTRAIN_NULL_SESSION_WITH_BEH,
    'stitching_for_imposter_session': STITCHING_FOR_IMPOSTER_SESSION,
    'use_imposter_session_for_balancing': USE_IMPOSTER_SESSION_FOR_BALANCING,
    'imposter_generate_fake': IMPOSTER_GENERATE_FAKE,
    'filter_pseudosessions_on_mutualInformation': FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION,
    'max_number_trials_when_no_stitching_for_imposter_session': MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION,
    # MISC
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE,
    'save_predictions': SAVE_PREDICTIONS,
    'save_predictions_pseudo': SAVE_PREDICTIONS_PSEUDO,
    'save_binned': SAVE_BINNED,
    'binarization_value': BINARIZATION_VALUE,
    'add_to_saving_path': ADD_TO_SAVING_PATH,
    'motor_regressors': MOTOR_REGRESSORS,
    'motor_regressors_only': MOTOR_REGRESSORS_ONLY,
    'imposterdf': None,
}
