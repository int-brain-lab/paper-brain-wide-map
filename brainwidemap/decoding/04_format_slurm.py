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
                       'mask': ''.join([str(item) for item in list(result['fit'][i_run]['mask'].values * 1)]),
                       'R2_test': result['fit'][i_run]['Rsquared_test_full'],
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

'''
resultsdf = resultsdf[resultsdf.subject == 'NYU-12']
resultsdf = resultsdf[resultsdf.eid == 'a8a8af78-16de-4841-ab07-fde4b5281a03']
resultsdf.region = resultsdf.region.apply(lambda x:x[0])
resultsdf = resultsdf[resultsdf.region == 'CA1']
resultsdf = resultsdf[resultsdf.probe == 'probe00']
resultsdf = resultsdf[resultsdf.run_id == 1]
subdf = resultsdf.set_index(['subject', 'eid', 'probe', 'region']).drop('fold', axis=1)

estimatorstr = [estimator_strs[i] for i in range(len(estimator_options)) if ESTIMATOR == estimator_options[i]]
assert len(estimatorstr)==1
estimatorstr = estimatorstr[0] 
start_tw, end_tw = TIME_WINDOW   
    
model_str = 'interIndividual' if isinstance(MODEL, str) else modeldispatcher[MODEL]

fn = str(RESULTS_DIR.joinpath('decoding','results','neural','ephys', 
                              '_'.join([DATE, 'decode', TARGET,
                               model_str if TARGET in ['prior','pLeft'] else 'task',
                               estimatorstr, 
                               'align', ALIGN_TIME, 
                               str(N_PSEUDO), 'pseudosessions', 
                               'regionWise' if SINGLE_REGION else 'allProbes',
                               'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_')])))
if ADD_TO_SAVING_PATH != '':
    fn = fn + '_' + ADD_TO_SAVING_PATH
'''

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
