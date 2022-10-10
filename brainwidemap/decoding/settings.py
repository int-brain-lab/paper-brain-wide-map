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


# DEFINE SOME OPTION LISTS
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


DATE = '10-10-2022'


# SELECT DECODING TARGET
# --------------------------------------------------
# single-bin targets
# - pLeft
# - signcont
# - choice
# - feedback
#
# multi-bin targets
# - wheel-vel
# - wheel-speed
# - pupil
# - l-paw-pos
# - l-paw-vel
# - l-paw-speed
# - l-whisker-me
# - r-paw-pos
# - r-paw-vel
# - r-paw-speed
# - r-whisker-me
TARGET = 'wheel-speed'

# model of prior
# - expSmoothing_prevAction
# - expSmoothing_stimside
# - optBay
# - oracle (experimenter-defined 0.2/0.8)
# - absolute path; this will be the interindividual results
MODEL = None

# if true, trains the behavioral model session-wise else mouse-wise
BEH_MOUSELEVEL_TRAINING = False


# TIME WINDOW PARAMS
# --------------------------------------------------
# event on which windows are aligned
# - firstMovement_times
# - goCue_times
# - stimOn_times
# - feedback_times
ALIGN_TIME = 'firstMovement_times'

# start and end of decoding window, relative to ALIGN_TIME (seconds)
TIME_WINDOW = (-0.2, 1.0)

# size of individual bins within time window (seconds)
BINSIZE = 0.02

# number of bins to use for prediction
N_BINS_LAG = 10


# DECODER PARAMS
# --------------------------------------------------
# sklearn model
# - linear
# - lasso (linear + L1)
# - ridge (linear + L2)
# - logistic
ESTIMATOR = 'lasso'

# default args for decoder
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 20000, 'fit_intercept': True}

# hyperparameter values to search over
if ESTIMATOR == 'logistic':
    HPARAM_GRID = {'C': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
else:
    HPARAM_GRID = {'alpha': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}

# number of pseudo/imposter sessions to fit per session
N_PSEUDO = 4

# number of pseudo/imposter sessions to assign per cluster job
N_PSEUDO_PER_JOB = 2

# number of times to repeat full nested xv with different folds
N_RUNS = 2

# true for interleaved xv, false for contiguous
SHUFFLE = True

# if TRUE, decoding is launched in a quasi-random, reproducible way => it sets the seed
QUASI_RANDOM = False

# seems to work better with BALANCED_WEIGHT=False, but putting True is important
BALANCED_WEIGHT = False
BALANCED_CONTINUOUS_TARGET = True  # is target continuous or discrete FOR BALANCED WEIGHTING


# CLUSTER/UNIT PARAMS
# --------------------------------------------------
# regions with units below this threshold are skipped
MIN_UNITS = 10

# fraction of qc criteria each unit needs to pass for inclusion
QC_CRITERIA = 3 / 3

# perform decoding on region-wise or whole-brain decoding
SINGLE_REGION = True

# merge probes before performing analysis
MERGED_PROBES = False


# SESSION/BEHAVIOR PARAMS
# --------------------------------------------------
# minimum number of behavioral trials completed by subject
MIN_BEHAV_TRIAS = 200

# remove trials with reaction times below/above these values (seconds)
MIN_RT = 0.08  # 0.08  # Float (s) or None
MAX_RT = 2.0

# remove trials with durations below/above these values (seconds)
MIN_LEN = None
MAX_LEN = None

# true to take out unbiased block at beginning of session
NO_UNBIAS = False

# number of trials to remove from the end of the session
N_TRIALS_TAKEOUT_END = 0


# NULL DISTRIBUTION PARAMS
# --------------------------------------------------
# false will use pseudo-sessions to create null distributions
USE_IMPOSTER_SESSION = True

# true to just use ephys sessions, false to use training sessions (more templates)
IMPOSTER_GENERATE_FROM_EPHYS = False

# TODO
CONSTRAIN_NULL_SESSION_WITH_BEH = False

# if true, stitches sessions to create imposters
STITCHING_FOR_IMPOSTER_SESSION = True

# if false, simulate the model (should be false)
USE_IMPOSTER_SESSION_FOR_BALANCING = False

# testing purposes?
IMPOSTER_GENERATE_FAKE = False

FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION = False

# this is a constraint on the number of trials of a session to insure that there will be at least
# 1000 unstitched imposter sessions. IMPORTANT, with this number, you can not generate more than
# 1000 control imposter sessions
MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION = 700


# MISC PARAMS
# --------------------------------------------------
# uses openturns to perform kernel density estimation
USE_OPENTURNS = False
BIN_SIZE_KDE = 0.05

SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_PSEUDO = False
SAVE_BINNED = False

# to binarize the target -> could be useful with logistic regression estimator
BINARIZATION_VALUE = None

# DLC
MOTOR_REGRESSORS = False
MOTOR_REGRESSORS_ONLY = False  # only use motor regressors


# RUN CHECKS
# --------------------------------------------------

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
