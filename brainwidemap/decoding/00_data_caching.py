from pathlib import Path

from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import (
    bwm_query, load_good_units, load_trials_and_mask, filter_regions, filter_sessions,
    download_aggregate_tables)

# For caching, we use an online instance. You can choose the cache directory if you want
one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')

# Get the dataframe of all included sessions and probe insertions, use frozen query
bwm_df = bwm_query(freeze='2022_10_initial')

# Download the latest clusters table, we use the same cache as above
clusters_table = download_aggregate_tables(one, type='clusters', target_path=cache_dir)
# Map probes to regions (here: Beryl mapping) and filter according to QC, number of units and probes per region
region_df = filter_regions(bwm_df['pid'], clusters_table=clusters_table, mapping='Beryl', min_qc=1,
                           min_units_region=10, min_probes_region=2)

# Download the latest trials table and filter for sessions that have at least 200 trials fulfilling BWM criteria
trials_table = download_aggregate_tables(one, type='trials', target_path=cache_dir)
eids = filter_sessions(bwm_df['eid'], trials_table=trials_table, min_trials=200)

# Remove probes and sessions based on those filters
bwm_df = bwm_df[bwm_df['pid'].isin(region_df['pid'].unique())]
bwm_df = bwm_df[bwm_df['eid'].isin(eids)]

# Download cluster information and spike trains for all good units.
for pid in bwm_df['pid']:
    print(f"Downloading spike sorting data for {pid}")
    spikes, clusters = load_good_units(one, pid, compute_metrics=True)

# Download trials for all sessions
for eid in bwm_df['eid']:
    print(f"Downloading trials data for {eid}")
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_trials()

# Download wheel data for all sessions (there is currently one error so skipt that session, should be fixed tomorrow)
for eid in bwm_df['eid']:
    if eid == 'cc45c568-c3b9-4f74-836e-c87762e898c8':
        continue
    else:
        print(f"Downloading wheel data for {eid}")
        sess_loader = SessionLoader(one, eid)
        sess_loader.load_wheel()

# Download whisker data for all sessions
me_err = []
for eid in bwm_df['eid']:
    try:
        print(f"Downloading motion energy data for {eid}")
        sess_loader = SessionLoader(one, eid)
        sess_loader.load_motion_energy(views=['left', 'right'])
    except BaseException as e:
        print(eid, e)
        me_err.append((eid, e))


