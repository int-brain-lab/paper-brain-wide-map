import os
import numpy as np
import pandas as pd
from brainwidemap.decoding.settings import *
from brainwidemap.decoding.functions.process_outputs import create_pdtable_from_raw

score_name = 'balanced_acc_test' if estimatorstr=='LogisticsRegression' else 'R2_test'

print(f'Working on {DATE} {TARGET}')
file_pre = SETTINGS_FORMAT_NAME

MIN_SESS_PER_REG = 1

# aggregate parallelized formatting
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_pickle(file_pre+'_paraindex'+str(i)+'.pkl')
    res = pd.concat([res, res_new], axis=0)
#res = res.loc[res['eid']=='02fbb6da-3034-47d6-a61b-7d06c796a830']
res = res.reset_index()

# create summary table 
print('creating pdtable')
res_table, xy_table = create_pdtable_from_raw(res, 
                                    score_name=score_name,
                                    N_PSEUDO=N_PSEUDO,
                                    N_RUN=N_RUNS,
                                    RETURN_X_Y=True,
                                    SCALAR_PER_TRIAL=False if TARGET in ['wheel-vel', 
                                                                         'wheel-speed', 
                                                                         'l-whisker-me', 
                                                                         'r-whisker-me'] else True,
                                    SAVE_REGRESSORS=False if TARGET in ['wheel-vel',
                                                                        'wheel-speed',
                                                                        'l-whisker-me',
                                                                        'r-whisker-me'] else True)

# filter for valide regions, those that have at least two eids
valid_reg = np.array([len(res_table.loc[res_table['region']==reg])>=MIN_SESS_PER_REG for reg in res_table['region']])
res_table = res_table.loc[valid_reg]
xy_table = xy_table.loc[valid_reg]

# save results
save_dir = RESULTS_DIR.joinpath("decoding", "results", "summary")
print('saved dir is ', str(save_dir))
if not os.path.exists(str(save_dir)):
    os.makedirs(str(save_dir))
SAVE_SUMMARY_PATH = str(save_dir.joinpath(os.path.basename(SETTINGS_FORMAT_NAME)))
res_table.to_csv(SAVE_SUMMARY_PATH + '.csv')
xy_table.to_pickle(SAVE_SUMMARY_PATH + '_xy.pkl')
