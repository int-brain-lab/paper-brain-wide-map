from brainbox.io.one import SessionLoader
from one.api import ONE
import sys

from brainwidemap import bwm_query, load_good_units

# number of jobs which should be submitted by a .sh file to parallelize data caching
# data is cached in N_PARA parallel jobs and para_index indicates which parallel job this is
N_PARA = 50
# para_index should be in [0, N_PARA-1]
para_index = int(sys.argv[1]) - 1

"""
-----------
USER INPUTS
-----------
"""

# Whether to download wheel and whisker data as well
wheel_data = True
whisker_data = False
# You can choose the cache directory if you want it to be different from the default
cache_dir = None  # Path('/full/path/to/cache')

"""
----------
CACHE DATA
-----------
"""
# For caching, we use an online instance.
if cache_dir:
    one = ONE(cache_dir=cache_dir,
              base_url='https://openalyx.internationalbrainlab.org',
              mode='remote')
else:
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              mode='remote')
    cache_dir = one.cache_dir

BAD_PIDS = []  # can exclude a list of PIDS here, but currently not necessary
BAD_EIDS = [one.pid2eid(p)[0] for p in BAD_PIDS]

# Get the dataframe of all included sessions and probe insertions, use frozen query
bwm_df = bwm_query(freeze='2022_10_bwm_release')


# Download cluster information and spike trains for all good units.
for count, pid in enumerate(bwm_df['pid']):
    if not (count % N_PARA == para_index):
        continue
    elif pid in BAD_PIDS:
        continue
    else:
        print(f"Downloading spike sorting data for {pid}")
        try:
            spikes, clusters = load_good_units(one, pid, compute_metrics=False)
        except Exception as e:
            print(e)
            print(f"Downloading failed for spike sorting data, pid {pid}, skipping")

# Download trials for all sessions
for count, eid in enumerate(bwm_df['eid']):
    if not (count % N_PARA == para_index):
        continue
    else:  # Oddly not problem with errors here, dont need BAD_EIDS like wheel or spike sorting
        print(f"Downloading trials data for {eid}")
        try:
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_trials()
        except Exception as e:
            print(e)
            print(f"Downloading failed for trials data, eid {eid}, skipping")

# Download wheel data for all sessions
if wheel_data:
    for count, eid in enumerate(bwm_df['eid']):
        if not (count % N_PARA == para_index):
            continue
        elif eid in BAD_EIDS:
            continue
        else:
            print(f"Downloading wheel data for {eid}")
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_wheel()

# Download whisker data for all sessions
if whisker_data:
    me_err = []
    for count, eid in enumerate(bwm_df['eid']):
        if not (count % N_PARA == para_index):
            continue
        elif eid in BAD_EIDS:
            continue
        else:
            print(f"Downloading motion energy data for {eid}")
            sess_loader = SessionLoader(one=one, eid=eid)
            sess_loader.load_motion_energy(views=['left', 'right'])
