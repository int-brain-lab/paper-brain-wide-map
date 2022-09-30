from braindelphi.decoding.functions.nulldistributions import generate_null_distribution_session
from behavior_models.models.utils import format_data as format_data_mut
from behavior_models.models.utils import format_input as format_input_mut
from braindelphi.params import CACHE_PATH
from braindelphi.decoding.functions.utils import load_metadata
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from braindelphi.decoding.functions.process_targets import check_bhv_fit_exists
from braindelphi.decoding.settings import modeldispatcher
import pickle
from pathlib import Path
import numpy as np

# load a trials_df
bwmdf, _ = load_metadata(CACHE_PATH.joinpath('*_%s_metadata.pkl' % 'ephys').as_posix())
pid = bwmdf['dataset_filenames'].iloc[0]
trials_df = pickle.load(open(pid.reg_file, 'rb'))['trials_df']
eid = '56956777-dca5-468c-87cb-78150432cc57'
subject_name = 'NYU-11'

# folder in which the behavioral models will be stored
BEH_MOD_PATH = Path('sandbox/behavioral')
BEH_MOD_PATH.mkdir(exist_ok=True, parents=True)

# script to generate frankenstein session
metadata = {
     'eids_train': np.array([eid]),
     'model': expSmoothing_prevAction,
     'behfit_path': BEH_MOD_PATH,
     'subject': subject_name,
     'use_imposter_session': False,
     'filter_pseudosessions_on_mutualInformation': True,
     'constrain_null_session_with_beh': False,
     'modeldispatcher': modeldispatcher,
     'model_parameters': None,
}

side, stim, act, _ = format_data_mut(trials_df)
stimuli, actions, stim_side = format_input_mut([stim], [act], [side])
behmodel = expSmoothing_prevAction(metadata['behfit_path'], metadata['eids_train'], metadata['subject'],
                           actions, stimuli, stim_side, single_zeta=True)
istrained, _ = check_bhv_fit_exists(metadata['subject'], metadata['model'], metadata['eids_train'],
                                    metadata['behfit_path'], modeldispatcher=modeldispatcher,
                                    single_zeta=True)
if not istrained:
    behmodel.load_or_train(remove_old=False)

frankenstein = generate_null_distribution_session(trials_df, metadata, **metadata)