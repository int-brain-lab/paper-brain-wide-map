import logging
import numpy as np
from brainwidemap.decoding.functions.process_targets import optimal_Bayesian
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
import sklearn.linear_model as sklm
import warnings

logger = logging.getLogger('ibllib')
logger.disabled = True

NEURAL_DTYPE = 'ephys'  # 'ephys' or 'widefield'
DATE = '27-09-2022'  # date 12 prev, 13 next, 14 prev

# aligned -> histology was performed by one experimenter 
# resolved -> histology was performed by 2-3 experiments
SESS_CRITERION = 'resolved-behavior'  # aligned and behavior
ALIGN_TIME = 'stimOn_times'
TARGET = 'pLeft'  # 'signcont' or 'pLeft'
if TARGET not in ['pLeft', 'signcont', 'strengthcont', 'choice', 'feedback']:
    raise ValueError('TARGET can only be pLeft, signcont, strengthcont, choice or feedback')
# NB: if TARGET='signcont', MODEL with define how the neurometric curves will be generated. else MODEL computes TARGET
# if MODEL is a path, this will be the interindividual results
MODEL = expSmoothing_prevAction  # 'population_level_Nmice101_NmodelsClasses7_processed.pkl' #expSmoothing_stimside, expSmoothing_prevAction, optimal_Bayesian or None(=Oracle)
BEH_MOUSELEVEL_TRAINING = False  # if True, trains the behavioral model session-wise else mouse-wise
TIME_WINDOW = (-0.6, -0.1)  # (0, 0.1)  #
ESTIMATOR = sklm.Lasso  # Must be in keys of strlut above
BINARIZATION_VALUE = None  # to binarize the target -> could be useful with logistic regression estimator
ESTIMATOR_KWARGS = {'tol': 0.0001, 'max_iter': 20000, 'fit_intercept': True}
N_PSEUDO = 200
N_PSEUDO_PER_JOB = 10
N_JOBS_PER_SESSION = N_PSEUDO // N_PSEUDO_PER_JOB
N_RUNS = 10
MIN_UNITS = 10
NB_TRIALS_TAKEOUT_END = 0
MIN_BEHAV_TRIAS = 150 if NEURAL_DTYPE == 'ephys' else 150  # default BWM setting is 400. 200 must remain after filtering
MIN_RT = 0.08  # 0.08  # Float (s) or None
MAX_RT = None
SINGLE_REGION = True  # perform decoding on region-wise or whole brain analysis
MERGED_PROBES = True  # merge probes before performing analysis
NO_UNBIAS = False  # take out unbiased trials
SHUFFLE = True  # interleaved cross validation
BORDER_QUANTILES_NEUROMETRIC = [.3, .7]  # [.3, .4, .5, .6, .7]
COMPUTE_NEUROMETRIC = False
FORCE_POSITIVE_NEURO_SLOPES = False
SAVE_PREDICTIONS = True

# Basically, quality metric on the stability of a single unit. Should have 1 metric per neuron
QC_CRITERIA = 3 / 3  # 3 / 3  # In {None, 1/3, 2/3, 3/3}
NORMALIZE_INPUT = False  # take out mean of the neural activity per unit across trials
NORMALIZE_OUTPUT = False  # take out mean of output to predict
if NORMALIZE_INPUT or NORMALIZE_OUTPUT:
    warnings.warn('This feature has not been tested')
USE_IMPOSTER_SESSION = False  # if false, it uses pseudosessions and simulates the model when action are necessary
FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION = False
STITCHING_FOR_IMPORTER_SESSION = False  # if true, stitches sessions to create importers
MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPORTER_SESSION = 700  # this is a constraint on the number of trials of a session
# to insure that there will be at least 1000 unstitched imposter sessions. IMPORTANT, with this number, you can not
# generate more than 1000 control imposter sessions
CONSTRAIN_NULL_SESSION_WITH_BEH = False  # add behavioral constraints
USE_IMPOSTER_SESSION_FOR_BALANCING = False  # if false, it simulates the model (should be False)
SIMULATE_NEURAL_DATA = False
QUASI_RANDOM = False  # if TRUE, decoding is launched in a quasi-random, reproducible way => it sets the seed

BALANCED_WEIGHT = False  # seems to work better with BALANCED_WEIGHT=False, but putting True is important
BALANCED_CONTINUOUS_TARGET = True  # is target continuous or discrete FOR BALANCED WEIGHTING
USE_OPENTURNS = False  # uses openturns to perform kernel density estimation
BIN_SIZE_KDE = 0.05  # size of the kde bin
HPARAM_GRID = ({
    #'alpha': np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
     'alpha': np.array([0.001, 0.01, 0.1]) #lasso
} if not (sklm.LogisticRegression == ESTIMATOR) else {
    'C': np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
})
SAVE_BINNED = False  # Debugging parameter, not usually necessary
COMPUTE_NEURO_ON_EACH_FOLD = False  # if True, expect a script that is 5 times slower
ADD_TO_SAVING_PATH = (
    'imposterSess_%i_balancedWeight_%i_RegionLevel_%i_mergedProbes_%i_behMouseLevelTraining_%i_simulated_%i_constrainNullSess_%i'
    % (USE_IMPOSTER_SESSION, BALANCED_WEIGHT, SINGLE_REGION, MERGED_PROBES,
       BEH_MOUSELEVEL_TRAINING, SIMULATE_NEURAL_DATA, CONSTRAIN_NULL_SESSION_WITH_BEH))

# WIDE FIELD IMAGING
WFI_HEMISPHERES = ['left', 'right']  # 'left' and/or 'right'
WFI_NB_FRAMES_START = -2  # left signed number of frames from ALIGN_TIME (frame included)
WFI_NB_FRAMES_END = -2  # right signed number of frames from ALIGN_TIME (frame included). If 0, the align time frame is included
WFI_AVERAGE_OVER_FRAMES = False

if NEURAL_DTYPE == 'widefield' and WFI_NB_FRAMES_START > WFI_NB_FRAMES_END:
    raise ValueError('there is a problem in the specification of the timing of the widefield')

# WHEEL VELOCITY
MIN_LEN = None  # min length of trial
MAX_LEN = None  # max length of trial

# DEEPLABCUT MOVEMENT REGRESSORS
MOTOR_REGRESSORS = False
MOTOR_REGRESSORS_ONLY = False # only _use motor regressors

# DO WE WANT TO DECODE THE PREVIOUS CONTRAST ? (FOR DEBUGGING)
DECODE_PREV_CONTRAST = False

# DO WE WANT TO DECODE THE DERIVATIVE OF THE TARGET SIGNAL ?
DECODE_DERIVATIVE = False


# session to be excluded (by Olivier Winter)
excludes = [
    'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',  # inconsistent trials object: relaunched task on 31-12-2021
    'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',  # same same
    '7a887357-850a-4378-bd2a-b5bc8bdd3aac',  # same same
    '56b57c38-2699-4091-90a8-aba35103155e',  # load object pickle error
    '09394481-8dd2-4d5c-9327-f2753ede92d7',  # same same
]

modeldispatcher = {
    expSmoothing_prevAction: expSmoothing_prevAction.name,
    expSmoothing_stimside: expSmoothing_stimside.name,
    optimal_Bayesian: 'optBay',
    None: 'oracle'
}

strlut = {sklm.Lasso: "Lasso",
          sklm.LassoCV: "LassoCV",
          sklm.Ridge: "Ridge",
          sklm.RidgeCV: "RidgeCV",
          sklm.LinearRegression: "PureLinear",
          sklm.LogisticRegression: "Logistic"}

if TARGET in ['choice', 'feedback'] and (MODEL != expSmoothing_prevAction or USE_IMPOSTER_SESSION):
    raise ValueError('if you want to decode choice or feedback, you must use the actionKernel model and frankenstein sessions')

if USE_IMPOSTER_SESSION and COMPUTE_NEUROMETRIC:
    raise ValueError('you can not use imposter sessions if you want to to compute the neurometric')

if USE_IMPOSTER_SESSION_FOR_BALANCING:
    raise ValueError('this is not implemented yet -- or it is but unsure of the state given recent code featuring')

if ESTIMATOR == sklm.LogisticRegression and BALANCED_CONTINUOUS_TARGET:
    raise ValueError('you can not have a continuous target with logistic regression')

# ValueErrors and NotImplementedErrors
if not SINGLE_REGION and not MERGED_PROBES:
    raise ValueError('full probes analysis can only be done with merged probes')

if MODEL not in list(modeldispatcher.keys()) and not isinstance(MODEL, str):
    raise NotImplementedError('this MODEL is not supported yet')

if COMPUTE_NEUROMETRIC and TARGET != 'signcont':
    raise ValueError('the target should be signcont to compute neurometric curves')

if len(BORDER_QUANTILES_NEUROMETRIC) == 0 and MODEL is not None:
    raise ValueError('BORDER_QUANTILES_NEUROMETRIC must be at least of 1 when MODEL is specified')

if len(BORDER_QUANTILES_NEUROMETRIC) != 0 and MODEL is None:
    raise ValueError(
        'BORDER_QUANTILES_NEUROMETRIC must be empty when MODEL is not specified - oracle pLeft used'
    )


fit_metadata = {
    'date': DATE,
    'criterion': SESS_CRITERION,
    'target': TARGET,
    'model_type': 'inter_individual' if MODEL not in modeldispatcher.keys() else modeldispatcher[MODEL],
    'align_time': ALIGN_TIME,
    'time_window': TIME_WINDOW,
    'estimator': ESTIMATOR,
    'nb_runs': N_RUNS,
    'n_pseudo': N_PSEUDO,
    'min_units': MIN_UNITS,
    'min_behav_trials': MIN_BEHAV_TRIAS,
    'min_rt': MIN_RT,
    'qc_criteria': QC_CRITERIA,
    'shuffle': SHUFFLE,
    'no_unbias': NO_UNBIAS,
    'hyperparameter_grid': HPARAM_GRID,
    'save_binned': SAVE_BINNED,
    'balanced_weight': BALANCED_WEIGHT,
    'force_positive_neuro_slopes': FORCE_POSITIVE_NEURO_SLOPES,
    'compute_neurometric': COMPUTE_NEUROMETRIC,
    'n_runs': N_RUNS,
    'normalize_output': NORMALIZE_OUTPUT,
    'normalize_input': NORMALIZE_INPUT,
    'single_region': SINGLE_REGION,
    'use_imposter_session': USE_IMPOSTER_SESSION,
    'balanced_continuous_target': BALANCED_CONTINUOUS_TARGET,
    'use_openturns': USE_OPENTURNS,
    'bin_size_kde': BIN_SIZE_KDE,
    'use_imposter_session_for_balancing': USE_IMPOSTER_SESSION_FOR_BALANCING,
    'beh_mouseLevel_training': BEH_MOUSELEVEL_TRAINING,
    'simulate_neural_data': SIMULATE_NEURAL_DATA,
    'constrain_null_session_with_beh': CONSTRAIN_NULL_SESSION_WITH_BEH,
    'neural_dtype': NEURAL_DTYPE,
    'modeldispatcher': modeldispatcher,
    'estimator_kwargs': ESTIMATOR_KWARGS,
    'hyperparam_grid': HPARAM_GRID,
    'add_to_saving_path': ADD_TO_SAVING_PATH,
    'min_len': MIN_LEN,
    'max_len': MAX_LEN,
    'save_predictions': SAVE_PREDICTIONS,
    'wfi_nb_frames_start': WFI_NB_FRAMES_START,
    'wfi_nb_frames_end': WFI_NB_FRAMES_END,
    'quasi_random': QUASI_RANDOM,
    'motor_regressors':MOTOR_REGRESSORS,
    'motor_regressors_only':MOTOR_REGRESSORS_ONLY,
    'decode_prev_contrast':DECODE_PREV_CONTRAST,
    'decode_derivative':DECODE_DERIVATIVE
}

if NEURAL_DTYPE == 'widefield':
    fit_metadata['wfi_hemispheres'] = WFI_HEMISPHERES
    fit_metadata['wfi_nb_frames'] = WFI_HEMISPHERES

kwargs = {
    'date': DATE,
    'nb_runs': N_RUNS,
    'single_region': SINGLE_REGION,
    'merged_probes': MERGED_PROBES,
    'modeldispatcher': modeldispatcher,
    'estimator_kwargs': ESTIMATOR_KWARGS,
    'hyperparam_grid': HPARAM_GRID,
    'save_binned': SAVE_BINNED,
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
    'wfi_nb_frames_start': WFI_NB_FRAMES_START,
    'wfi_nb_frames_end': WFI_NB_FRAMES_END,
    'quasi_random': QUASI_RANDOM,
    'nb_trials_takeout_end': NB_TRIALS_TAKEOUT_END,
    'stitching_for_imposter_session': STITCHING_FOR_IMPORTER_SESSION,
    'max_number_trials_when_no_stitching_for_imposter_session': MAX_NUMBER_TRIALS_WHEN_NO_STITCHING_FOR_IMPORTER_SESSION,
    'filter_pseudosessions_on_mutualInformation': FILTER_PSEUDOSESSIONS_ON_MUTUALINFORMATION,
    'motor_regressors':MOTOR_REGRESSORS,
    'motor_regressors_only':MOTOR_REGRESSORS_ONLY,
    'decode_prev_contrast':DECODE_PREV_CONTRAST,
    'decode_derivative':DECODE_DERIVATIVE,
    'wfi_average_over_frames': WFI_AVERAGE_OVER_FRAMES,
    'imposterdf': None
}
