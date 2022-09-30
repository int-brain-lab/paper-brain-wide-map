import pickle
from behavior_models.models.utils import format_data as format_data_mut
import pandas as pd
import glob
from braindelphi.decoding.settings import *
import models.utils as mut
from braindelphi.params import FIT_PATH
from braindelphi.decoding.settings import modeldispatcher
from tqdm import tqdm

SAVE_KFOLDS = False


decoding_dates = ['40-06-2022','26-06-2022','27-06-2022','28-06-2022',
                   '29-06-2022','30-06-2022','31-06-2022','32-06-2022',
                   '33-06-2022','34-06-2022','41-06-2022']

decoding_dates_region_level = ['01-07-2022','02-07-2022','03-07-2022','04-07-2022',
                   '05-07-2022','06-07-2022','07-07-2022','08-07-2022',
                   '09-07-2022','10-07-2022','11-07-2022']

decoding_frames = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

for i in range(len(decoding_dates_region_level)): #:

    date = decoding_dates_region_level[i]
    WFI_NB_FRAMES_START = decoding_frames[i]
    WFI_NB_FRAMES_END = decoding_frames[i]
    print(date,decoding_frames[i])

    '''
    i = 0
    date = '00-07-2022'
    WFI_NB_FRAMES_START = -2
    WFI_NB_FRAMES_END = -2
    '''

    finished = glob.glob(str(FIT_PATH.joinpath(kwargs['neural_dtype'], "*", "*", "*", "*%s*" % date)))
    print(len(finished))

    weight_indexers = ['subject', 'eid','probe','region'] 
    weightsdict = {}
    for fn in tqdm(finished):
        if 'pseudo_id_-1' in fn :
            fo = open(fn, 'rb')
            result = pickle.load(fo)
            fo.close()
            for i_run in range(len(result['fit'])):
                weightsdict = {**weightsdict, **{(tuple( (result[x][0] if isinstance(result[x], list) else result[x]) for x in weight_indexers)
                                                + (result['pseudo_id'],
                                                    i_run + 1))
                                                : np.vstack(result['fit'][i_run]['weights'])}}

    weights = pd.Series(weightsdict).reset_index()
    weights.columns=['subject','session','hemisphere','region','pseudo_id','run_id','weights'] #'region' before hem

    estimatorstr = strlut[ESTIMATOR]

    if NEURAL_DTYPE == 'ephys':
        start_tw, end_tw = TIME_WINDOW
    if NEURAL_DTYPE == 'widefield':
        start_tw = WFI_NB_FRAMES_START
        end_tw = WFI_NB_FRAMES_END   

    model_str = 'interIndividual' if isinstance(MODEL, str) else modeldispatcher[MODEL]
    fn = str(FIT_PATH.joinpath(kwargs['neural_dtype'], '_'.join([date, 'decode', TARGET,
                                                                model_str if TARGET in ['prior',
                                                                                                            'pLeft']
                                                                else 'task',
                                                                estimatorstr, 'align', ALIGN_TIME, str(N_PSEUDO),
                                                                'pseudosessions',
                                                                'regionWise' if SINGLE_REGION else 'allProbes',
                                                                'timeWindow', str(start_tw).replace('.', '_'),
                                                                str(end_tw).replace('.', '_')])))

    if ADD_TO_SAVING_PATH != '':
        fn = fn + '_' + ADD_TO_SAVING_PATH

    weights_fn =  fn + '.w.pkl'

    with open(weights_fn, 'wb') as f:
        pickle.dump(weights, f)

