import pickle
import re
from behavior_models.models.utils import format_data as format_data_mut
import pandas as pd
import glob
from brainwidemap.decoding.settings import *
from brainwidemap.decoding.functions.process_outputs import fix_pd_regions
import models.utils as mut
#from brainwidemap.decoding.settings import RESULTS_DIR
#from brainwidemap.decoding.settings import modeldispatcher
#from brainwidemap.decoding.settings import SETTINGS_FORMAT_NAME
from tqdm import tqdm
import sys

para_index = int(sys.argv[1])-1
N_PARA = 50
print('target is ',TARGET)
print('date is ',DATE)

SAVE_KFOLDS = False

finished = glob.glob(str(RESULTS_DIR.joinpath("decoding", 
                                           "results", 
                                           "neural", 
                                           "ephys", 
                                           "*", 
                                           "*", 
                                           "*", 
                                           "*%s*%s*" % (DATE, TARGET))))
print('nb files:',len(finished))

indexers = ['subject', 'eid', 'probe', 'region']
resultslist = []

failed_load = 0
tot_index = -1
for fn in tqdm(finished):
    tot_index += 1
    if not ((tot_index%N_PARA) == para_index):
        continue
    try:
        print('file ', fn)
        fo = open(fn, 'rb')
        result = pickle.load(fo)
        fo.close()
        if result['fit'] is None:
            continue
        
        for i_run in range(len(result['fit'])):
            if not re.match(".*pseudo_id_-1.*", fn):
                tmpdict = {**{x: result[x] for x in indexers},
                       'fold': -1,
                       'pseudo_id': result['pseudo_id'],
                       'run_id': i_run + 1,
                       'R2_test': result['fit'][i_run]['Rsquared_test_full']}
            else:
                side, stim, act, _ = format_data_mut(result["fit"][i_run]["df"])
                mask = result["fit"][i_run]["mask"]  # np.all(result["fit"][i_run]["target"] == stim[mask])
                mask_trials_and_targets = result["fit"][i_run]["mask_trials_and_targets"]
                mask_diagnostics = result["fit"][i_run]["mask_diagnostics"]
                cluster_uuids = result["fit"][i_run]["cluster_uuids"]
                #full_test_prediction = np.zeros(np.array(result["fit"][i_run]["target"]).size)
                full_test_prediction = np.zeros(np.array(result["fit"][i_run]["target"]).shape)
                for k in range(len(result["fit"][i_run]["idxes_test"])):
                    if len(full_test_prediction.shape) == 1:
                        full_test_prediction[result["fit"][i_run]['idxes_test'][k]] = result["fit"][i_run]['predictions_test'][k]
                    elif len(full_test_prediction.shape) == 2:
                        #print(i_run, result["fit"][i_run])
                        full_test_prediction[result["fit"][i_run]['idxes_test'][k], :] = result["fit"][i_run]['predictions_test'][k]
                    else:
                        raise IndexError('full_test_prediction is not an acceptable shape')
                # neural_act = np.sign(full_test_prediction)
                #perf_allcontrasts = (side.values[mask][neural_act != 0] == neural_act[neural_act != 0]).mean()
                #perf_allcontrasts_prevtrial = (side.values[mask][1:] == neural_act[:-1])[neural_act[:-1] != 0].mean()
                #perf_0contrasts = (side.values[mask] == neural_act)[(stim[mask] == 0) * (neural_act != 0)].mean()
                #nb_trials_act_is_0 = (neural_act == 0).mean()
                tmpdict = {**{x: result[x] for x in indexers},
                       'fold': -1,
                       'pseudo_id': result['pseudo_id'],
                       'N_units': result['N_units'],
                       'run_id': i_run + 1,
                       'mask': ''.join([str(item) for item in list(mask.values * 1)]),
                       'mask_trials_and_targets': mask_trials_and_targets,
                       'mask_diagnostics': mask_diagnostics,
                       'cluster_uuids': cluster_uuids,
                       'R2_test': result['fit'][i_run]['Rsquared_test_full'],
                       'idxes_test': result['fit'][i_run]['idxes_test'],
                       'weights': result['fit'][i_run]['weights'],
                       'params': result['fit'][i_run]['best_params'],
                       'intercepts': result['fit'][i_run]['intercepts'],
                       #'full_prediction': full_test_prediction,
                       'prediction': list(result["fit"][i_run]['predictions_test']),
                       'target': list(result["fit"][i_run]["target"]),
                       'regressors': result['fit'][i_run]['regressors'],
                       # 'perf_allcontrast': perf_allcontrasts,
                       # 'perf_allcontrasts_prevtrial': perf_allcontrasts_prevtrial,
                       # 'perf_0contrast': perf_0contrasts,
                       # 'nb_trials_act_is_0': nb_trials_act_is_0,
                       }
            if 'acc_test_full' in result['fit'][i_run].keys():
                tmpdict = {**tmpdict, 'acc_test': result['fit'][i_run]['acc_test_full'],
                           'balanced_acc_test': result['fit'][i_run]['balanced_acc_test_full']}
            resultslist.append(tmpdict)

            if SAVE_KFOLDS:
                for kfold in range(result['fit'][i_run]['nFolds']):
                    tmpdict = {**{x: result[x] for x in indexers},
                               'fold': kfold,
                               'pseudo_id': result['pseudo_id'],
                               'N_units': result['N_units'],
                               'run_id': i_run + 1,
                               'R2_test': result['fit'][i_run]['Rsquareds_test'][kfold],
                               'Best_regulCoef': result['fit'][i_run]['best_params'][kfold],
                               }
                    resultslist.append(tmpdict)
    except EOFError:
        failed_load += 1
        pass
print('loading of %i files failed' % failed_load)
resultsdf = pd.DataFrame(resultslist)
resultsdf = fix_pd_regions(resultsdf)

fn = SETTINGS_FORMAT_NAME

fn = fn + '_paraindex' + str(para_index) + '.pkl'

metadata_df = pd.Series({'filename': fn,  'date': DATE, **params})
metadata_fn = '.'.join([fn.split('.')[0], 'metadata', 'pkl'])
print('saving pickle')
resultsdf.to_pickle(fn)
print('pickle saved')
print('saving metadata')
metadata_df.to_pickle(metadata_fn)
print('metadata saved')
