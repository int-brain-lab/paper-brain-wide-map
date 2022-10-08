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


DATE = '08-10-2022'


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
MODEL = 'expSmoothing_prevAction'

# add DLC data as additional regressors to neural activity
MOTOR_REGRESSORS = False
MOTOR_REGRESSORS_ONLY = False  # only use motor regressors

# do we want to decode the previous contrast? (FOR DEBUGGING)
DECODE_PREV_CONTRAST = False

# decode the derivative of the target signal
DECODE_DERIVATIVE = False


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
ESTIMATOR = 'ridge'

# default args for decoder
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 20000, 'fit_intercept': True}

# hyperparameter values to search over
if ESTIMATOR == 'logistic':
    HPARAM_GRID = {'C': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])}
else:
    HPARAM_GRID = {'alpha': np.array([0.001, 0.01, 0.1])}

# number of pseudo/imposter sessions to fit per session
N_PSEUDO = 200

# number of pseudo/imposter sessions to assign per cluster job
N_PSEUDO_PER_JOB = 10

# number of times to repeat full nested xv with different folds
N_RUNS = 10

# true for interleaved xv, false for contiguous
SHUFFLE = True

# data normalization
NORMALIZE_INPUT = False  # take out mean of the neural activity per unit across trials
NORMALIZE_OUTPUT = False  # take out mean of output to predict

BALANCED_WEIGHT = False  # seems to work better with BALANCED_WEIGHT=False, but putting True is important
BALANCED_CONTINUOUS_TARGET = True  # is target continuous or discrete FOR BALANCED WEIGHTING


# CLUSTER/UNIT PARAMS
# --------------------------------------------------
# type of neural data
# - ephys
# - widefield
NEURAL_DTYPE = 'ephys'

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
# aligned -> histology was performed by one experimenter
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'resolved-behavior'

# minimum number of behavioral trials completed by subject
MIN_BEHAV_TRIAS = 150 if NEURAL_DTYPE == 'ephys' else 150  # default BWM setting is 400. 200 must remain after filtering

# remove trials with reaction times below/above these values (seconds)
MIN_RT = 0.08  # 0.08  # Float (s) or None
MAX_RT = None

# min/max length of trials (seconds) (can be null)
MIN_LEN = None  # min length of trial
MAX_LEN = None  # max length of trial

# true to take out unbiased block at beginning of session
NO_UNBIAS = False

# number of trials to remove from the end of the session
NB_TRIALS_TAKEOUT_END = 0


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
SIMULATE_NEURAL_DATA = False

# uses openturns to perform kernel density estimation
USE_OPENTURNS = False
BIN_SIZE_KDE = 0.05

SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_PSEUDO = False
SAVE_BINNED = False

# if true, trains the behavioral model session-wise else mouse-wise
BEH_MOUSELEVEL_TRAINING = False

# to binarize the target -> could be useful with logistic regression estimator
BINARIZATION_VALUE = None

# if True, expect a script that is 5 times slower
COMPUTE_NEURO_ON_EACH_FOLD = False

# if TRUE, decoding is launched in a quasi-random, reproducible way => it sets the seed
QUASI_RANDOM = True


# NEUROMETRIC PARAMS
# --------------------------------------------------
BORDER_QUANTILES_NEUROMETRIC = [.3, .7]  # [.3, .4, .5, .6, .7]
COMPUTE_NEUROMETRIC = False
FORCE_POSITIVE_NEURO_SLOPES = False


# WIDEFIELD PARAMS
# --------------------------------------------------
WFI_HEMISPHERES = ['left', 'right']  # 'left' and/or 'right'
WFI_NB_FRAMES_START = -2  # left signed number of frames from ALIGN_TIME (frame included)
WFI_NB_FRAMES_END = -2  # right signed number of frames from ALIGN_TIME (frame included). If 0, the align time frame is included
WFI_AVERAGE_OVER_FRAMES = False


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

if NORMALIZE_INPUT or NORMALIZE_OUTPUT:
    warnings.warn('This feature has not been tested')

# ValueErrors and NotImplementedErrors
if NEURAL_DTYPE == 'widefield' and WFI_NB_FRAMES_START > WFI_NB_FRAMES_END:
    raise ValueError('there is a problem in the specification of the timing of the widefield')

if TARGET in ['choice', 'feedback'] and (MODEL != expSmoothing_prevAction or USE_IMPOSTER_SESSION):
    raise ValueError(
        f'if you want to decode choice or feedback, you must use the actionKernel model and '
        f'frankenstein sessions')

if USE_IMPOSTER_SESSION and COMPUTE_NEUROMETRIC:
    raise ValueError('you can not use imposter sessions if you want to to compute the neurometric')

if USE_IMPOSTER_SESSION_FOR_BALANCING:
    raise ValueError(
        f'this is not implemented yet -- or it is but unsure of the state given recent code '
        f'featuring')

if ESTIMATOR == sklm.LogisticRegression and BALANCED_CONTINUOUS_TARGET:
    raise ValueError('you can not have a continuous target with logistic regression')

if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('full probes analysis can only be done with merged probes')

if COMPUTE_NEUROMETRIC and TARGET != 'signcont':
    raise ValueError('the target should be signcont to compute neurometric curves')

if len(BORDER_QUANTILES_NEUROMETRIC) == 0 and MODEL is not None:
    raise ValueError('BORDER_QUANTILES_NEUROMETRIC must be at least of 1 when MODEL is specified')

if len(BORDER_QUANTILES_NEUROMETRIC) != 0 and MODEL is None:
    raise ValueError(
        f'BORDER_QUANTILES_NEUROMETRIC must be empty when MODEL is not specified '
        f'- oracle pLeft used'
    )

ADD_TO_SAVING_PATH = (
    'imposterSess_%i_balancedWeight_%i_RegionLevel_%i_mergedProbes_%i_behMouseLevelTraining_%i_simulated_%i_constrainNullSess_%i'
    % (USE_IMPOSTER_SESSION, BALANCED_WEIGHT, SINGLE_REGION, MERGED_PROBES,
       BEH_MOUSELEVEL_TRAINING, SIMULATE_NEURAL_DATA, CONSTRAIN_NULL_SESSION_WITH_BEH))


kwargs = {
    'date': DATE,
    'nb_runs': N_RUNS,
    'single_region': SINGLE_REGION,
    'merged_probes': MERGED_PROBES,
    'modeldispatcher': modeldispatcher,
    'estimator_kwargs': ESTIMATOR_KWARGS,
    'hyperparam_grid': HPARAM_GRID,
    'shuffle': SHUFFLE,
    'balanced_weight': BALANCED_WEIGHT,
    'normalize_input': NORMALIZE_INPUT,
    'normalize_output': NORMALIZE_OUTPUT,
    'compute_on_each_fold': COMPUTE_NEURO_ON_EACH_FOLD,
    'balanced_continuous_target': BALANCED_CONTINUOUS_TARGET,
    'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
    'estimator': ESTIMATOR,
    'target': TARGET,
    'model': MODEL,
    'binsize': BINSIZE,
    'n_bins_lag': N_BINS_LAG,
    'align_time': ALIGN_TIME,
    'no_unbias': NO_UNBIAS,
    'min_rt': MIN_RT,
    'max_rt': MAX_RT,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'qc_criteria': QC_CRITERIA,
    'min_units': MIN_UNITS,
    'time_window': TIME_WINDOW,
    'use_imposter_session': USE_IMPOSTER_SESSION,
    'compute_neurometric': COMPUTE_NEUROMETRIC,
    'border_quantiles_neurometric': BORDER_QUANTILES_NEUROMETRIC,
    'add_to_saving_path': ADD_TO_SAVING_PATH,
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE,
    'neural_dtype': NEURAL_DTYPE,
    'wfi_hemispheres': WFI_HEMISPHERES,
    'use_imposter_session_for_balancing': USE_IMPOSTER_SESSION_FOR_BALANCING,
    'beh_mouseLevel_training': BEH_MOUSELEVEL_TRAINING,
    'binarization_value': BINARIZATION_VALUE,
    'simulate_neural_data': SIMULATE_NEURAL_DATA,
    'constrain_null_session_with_beh': CONSTRAIN_NULL_SESSION_WITH_BEH,
    'min_len': MIN_LEN,
    'max_len': MAX_LEN,
    'save_predictions': SAVE_PREDICTIONS,
    'save_predictions_pseudo': SAVE_PREDICTIONS_PSEUDO,
    'save_binned': SAVE_BINNED,
    'wfi_nb_frames_start': WFI_NB_FRAMES_START,
    'wfi_nb_frames_end': WFI_NB_FRAMES_END,
    'quasi_random': QUASI_RANDOM,
    'nb_trials_takeout_end': NB_TRIALS_TAKEOUT_END,
    'stitching_for_imposter_session': STITCHING_FOR_IMPOSTER_SESSION,
    'max_number_trials_when_no_stitching_for_imposter_session': MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPOSTER_SESSION,
    'filter_pseudosessions_on_mutualInformation': FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION,
    'motor_regressors': MOTOR_REGRESSORS,
    'motor_regressors_only': MOTOR_REGRESSORS_ONLY,
    'decode_prev_contrast': DECODE_PREV_CONTRAST,
    'decode_derivative': DECODE_DERIVATIVE,
    'wfi_average_over_frames': WFI_AVERAGE_OVER_FRAMES,
    'imposterdf': None,
    'imposter_generate_from_ephys': IMPOSTER_GENERATE_FROM_EPHYS,
    'imposter_generate_fake': IMPOSTER_GENERATE_FAKE,
    'n_pseudo': N_PSEUDO,
    'n_pseudo_per_job': N_PSEUDO_PER_JOB,
}
