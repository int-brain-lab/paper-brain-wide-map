from one.api import ONE
from brainbox.io.one import SessionLoader

from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units

"""
Example 1:
==========
Load data for a single session from the released data set
"""
# Instantiate ONE
# If you DON'T want to download any new data, and just use previously cached data, set mode to 'local'
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')
# one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', mode='local')


# Get the dataframe of all released brainwide map sessions and probe insertions, use frozen query
bwm_df = bwm_query()

# Load spike sorting data for good units only
# The first time you run this, it will download the data and cache it locally. Subsequent runs will use the cached data.
pid = bwm_df.pid[10]
spikes, clusters = load_good_units(one, pid, compute_metrics=True)

# Load trials data and mask. Trials are excluded in the mask if reaction time is too long or too short,
# or if critical trial events are missing.
eid = bwm_df.eid[10]
trials, mask = load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default')


# To load other session data such as wheel, pose (DLC) and motion energy, we just use the SessionLoader, e.g.
sess_loader = SessionLoader(one, eid)
sess_loader.load_wheel(fs=1000)
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


"""
Example 2:
==========
Generate the table of all neurons, regions, probes and sessions used in the BWM paper analyses
"""

# Instantiate ONE, same as example 1
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')
# one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', mode='local')

# The bwm_units function downloads the relevant data tables and takes care of the filtering. The input parameters
# shown are the default values, which are used in the paper, but you can change them to suit your needs. 
unit_df = bwm_units(one, rt_range=(0.08, 0.2), min_errors=3, min_qc=1., min_units_sessions=(10, 2))

# rt_range -- the admissible range of trial length measured by goCue_time (start) and feedback_time (end)
# min_errors -- the minimum number of error trials per session after other criteria are applied
# min_qc -- the minimum quality criterion for a unit to be considered (between 0 and 1)
# min_units_sessions -- the first entry is the minimum of units per session per region for a session to be retained, the
#                       second entry is the minimum number of those sessions per region for a region to be retained
#                       (can be set to None to avoid this filter)
