from brainwidemap import bwm_query, load_good_units, load_trials_and_mask
from one.api import ONE
from brainbox.io.one import SessionLoader

# Instantiate ONE. If you DON'T want to download any new data, and just use previously cached data, set to local
# If you use the offline mode, make sure that you use this one instance in all functions that use one!
# Otherwise you run the risk of instantiating an online ONE later and downloading data anyway
one = ONE()
# one = ONE(mode='local')

# Get the dataframe of all included sessions and probe insertions
bwm_df = bwm_query(one)

# Load cluster information and spike trains for all good units
# The first time you run this, it will take a long time because it needs to download all data
for pid in bwm_df['pid']:
    spikes, clusters = load_good_units(one, pid)
    # If you want to not just download the data but actually use the loaded data, this is where you would
    # run your analysis code

# Load trials data and masks for all eids
for eid in bwm_df['eid'].unique():
    print(f'Loading trials for {eid}')
    trials, maks = load_trials_and_mask(one, eid)
    # Again, if you are not just interested in downloading the data but want to use it at this stage, your
    # analysis code would go here

# For other data types, such as wheel, pose (DLC), motion_energy and pupil just use the SessionLoader, e.g.
for eid in bwm_df['eid'].unique():
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_wheel()
    wheel = sess_loader.wheel

    # Or load all available data (trials, wheel, pose (DLC), motion_energy and pupil)
    sess_loader.load_session_data()
