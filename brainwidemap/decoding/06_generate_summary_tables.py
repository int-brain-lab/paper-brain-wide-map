import os
import numpy as np
import pandas as pd
from one.api import ONE
from brainwidemap.bwm_loading import bwm_units
from brainwidemap.decoding.settings import params, RESULTS_DIR, SETTINGS_FORMAT_NAME, estimatorstr
from brainwidemap.decoding.functions.process_outputs import create_pdtable_from_raw


score_name = 'balanced_acc_test' if estimatorstr == 'LogisticsRegression' else 'R2_test'

print(f'Working on {params["date"]} {params["target"]}')
file_pre = SETTINGS_FORMAT_NAME

CONT_TARGET = (params['target'] in ['wheel-vel', 'wheel-speed', 'l-whisker-me', 'r-whisker-me'])

# aggregate parallelized formatting
res = pd.DataFrame()
for i in range(50):
    res_new = pd.read_pickle(file_pre + '_paraindex' + str(i) + '.pkl')
    res = pd.concat([res, res_new], axis=0)
res = res.reset_index()

# create summary table
print('creating pdtable')
res_table, xy_table = create_pdtable_from_raw(
    res,
    score_name=score_name,
    N_PSEUDO=params['n_pseudo'],
    N_RUN=params['n_runs'],
    RETURN_X_Y=True,
    SCALAR_PER_TRIAL=False if CONT_TARGET else True,
    SAVE_REGRESSORS=False if CONT_TARGET else True
)


# Restrict to session-region pairs that are defined by the common function used to filter for valid units
one = ONE(base_url="https://openalyx.internationalbrainlab.org", mode='local')
units_df = bwm_units(one)
units_df['sessreg'] = units_df.apply(lambda x: f"{x['eid']}_{x['Beryl']}", axis=1)

res_table['sessreg'] = res_table.apply(lambda x: f"{x['eid']}_{x['region']}", axis=1)
res_table = res_table[res_table['sessreg'].isin(units_df['sessreg'])]
xy_table = xy_table.loc[xy_table['eid_region'].isin(units_df['sessreg'])]

# save results
save_dir = RESULTS_DIR.joinpath("decoding", "results", "summary")
print('saved dir is ', str(save_dir))
if not os.path.exists(str(save_dir)):
    os.makedirs(str(save_dir))
SAVE_SUMMARY_PATH = str(save_dir.joinpath(os.path.basename(SETTINGS_FORMAT_NAME)))
res_table.to_csv(SAVE_SUMMARY_PATH + '.csv')
xy_table.to_pickle(SAVE_SUMMARY_PATH + '_xy.pkl')
