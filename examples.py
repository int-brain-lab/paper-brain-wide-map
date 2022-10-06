from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions, \
    download_aggregate_tables

# Specify a path to download the cluster and trials tables
local_path = Path.home().joinpath('bwm_examples')
local_path.mkdir(exist_ok=True)

"""
Instantiate ONE
"""
# If you DON'T want to download any new data, and just use previously cached data, set mode to 'local'. Then make sure
# that you use this one instance in all functions that use ONE. Otherwise, you run the risk of instantiating an online
# ONE later and downloading data anyway
one = ONE()
# one = ONE(mode='local')

"""
Query for brainwide map probes and sessions
"""
# Get the dataframe of all included sessions and probe insertions, use frozen query
bwm_df = bwm_query(freeze='2022_10_initial')

"""
Map probes to regions and filter regions
"""
# First, download the latest clusters table
clusters_table = download_aggregate_tables(one, type='clusters', local_path=local_path)
# Map probes to regions (here: Beryl mapping) and filter according to QC, number of units per region and number of
# probes per region. These are the default values explicitly specified only for demonstration purposes.
region_df = filter_regions(bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl', min_qc=1,
                           min_units_region=10, min_probes_region=2)

"""
Filter sessions on min trial number
"""
# First, download the latest trials table. Note that currently, the bwm_include column is currently based on the
# whether a trial is included based on the default parameters of the load_trials_and_mask function (see below).
# If you need other criteria you can add a column to the trials table, or ask for help
trials_table = download_aggregate_tables(one, type='trials', local_path=local_path)
eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=200)


"""
Remove probes and sessions based on those filters
"""
bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]
bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

"""
Load spike sorting, good units only
"""
# Load cluster information and spike trains for all good units.
# The first time you run this, it will take a long time because it needs to download all data
for pid in bwm_df['pid']:
    spikes, clusters = load_good_units(one, pid, compute_metrics=True)
    # If you want to not just download the data but actually use the loaded data, this is where you would
    # run your analysis code

"""
Load and mask trials
"""
# Load trials data and masks for all eids. Trials are excluded in the mask if reaction time is too long or too short,
# or if certain events are NaN. Here we use the default parameters, specified explicitly only for demonstration purpose.
for eid in bwm_df['eid'].unique():
    print(f'Loading trials for {eid}')
    trials, mask = load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default')
    # Again, if you are not just interested in downloading the data but want to use it at this stage, your
    # analysis code would go here

"""
Load other session data
"""
# For other data types, such as wheel, pose (DLC) an motion energy, just use the SessionLoader, e.g.
for eid in bwm_df['eid'].unique():
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_wheel(sampling_rate=1000, smooth_size=0.03)
    # wheel is a dataframe that contains wheel times and position interpolated to a uniform sampling rate, velocity and
    # acceleration computed using Gaussian smoothing
    wheel = sess_loader.wheel

    sess_loader.load_pose(likelihood_thr=0.9, views=['left', 'right', 'body'])
    # pose is a dictionary of dataframes for each loaded camera (view), each containing times as well as x, y and
    # likelihood for each tracked point over time. Pose estimates in those dataframes are threshold at likelihood_thr
    # everything below is set to NaN
    dlc_left = sess_loader.pose['leftCamera']
    dlc_right = sess_loader.pose['rightCamera']
    dlc_body = sess_loader.pose['bodyCamera']

    # motion_energy is a dictionary of dataframes, each containing the times and the motion energy for each view
    # for the side views, they contain columns ['times', 'whiskerMotionEnergy'] for the body view it contains
    # ['times', 'bodyMotionEnergy']
    sess_loader.load_motion_energy(views=['left', 'right', 'body'])
    left_whisker = sess_loader.motion_energy['leftCamera']
    right_whisker = sess_loader.motion_energy['rightCamera']
    body = sess_loader.motion_energy['rightCamera']
