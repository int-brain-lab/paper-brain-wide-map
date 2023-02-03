"""
Loop over eids and load data needed to create impostor df
"""

import sys
import pandas as pd
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap import bwm_query
from brainwidemap.decoding.settings import params
from brainwidemap.decoding.settings import RESULTS_DIR

# Prepare where to store imposter sessions eid list if using biased choice world
decoding_dir = RESULTS_DIR.joinpath('decoding')
#decoding_dir.mkdir(exist_ok=True, parents=True)

# Cache data in N_PARA jobs in the 01_slurm*.sh file
N_PARA = 500
# PARAINDEX indicates which of the N_PARA jobs this is. PARAINDEX is in [0,N_PARA-1]
PARAINDEX = int(sys.argv[1])-1

if params['imposter_generate_from_ephys']:
    # ephys sessions from from one of 12 templates
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
    one.load_cache(clobber=True)
    bwm_df = bwm_query(freeze='2022_10_bwm_release')
    eids = bwm_df['eid'].unique()
else:
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    one.load_cache(clobber=True)
    eids = pd.read_parquet(decoding_dir.joinpath('imposter_behavior_sessions.pqt'))['eid']

for i, eid in enumerate(eids):
    if not (i % N_PARA == PARAINDEX):
        continue
    else:
        print('%i: %s' % (i, eid))
        try:
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_trials()
            sess_loader.load_wheel()
        except Exception as e:
            print('ERROR LOADING TRIALS DF')
            print(e)
            continue
