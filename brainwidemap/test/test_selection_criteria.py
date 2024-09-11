from one.api import ONE
from brainwidemap.bwm_loading import bwm_query, bwm_units


# code to run numbers https://github.com/int-brain-lab/ibldevtools/blob/master/julia/2022-10-19_bwm_numbers.py

# 139 subjects, 699 insertions, 459 sessions

#                                n_sessions  n_probes  n_units  n_regions
# Session and insertion QC              459       699   621733        281
# Session and insertion QC              459       699   621733        281
# Minimum 3 error trials                459       699   621733        281
# Single unit QC                        459       698    75708        268
# Gray matter regions                   459       698    65336        266
# Minimum 5 units per region            455       691    63357        245
# Minimum 2 sessions per region         454       690    62990        210
# Minimum 20 units per region           454       689    62857        201

def test_selection_criteria():
    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    rt_range = (0.08, 0.2)
    min_errors = 3
    min_qc = 1.
    min_units_sessions = (5, 2)

    bwm_df = bwm_query(one=one, freeze='2023_12_bwm_release')

    assert bwm_df.subject.nunique() == 139  # we have 139 subjects in the data release
    assert bwm_df.shape[0] == 699  # we have 699 insertions in the data release
    assert bwm_df.eid.nunique() == 459  # and 459 sessions

    # gray matter regions
    df_clusters = bwm_units(one=one, freeze='2023_12_bwm_release', rt_range=(0.08, 0.2), min_errors=3, min_qc=1, min_units_sessions=None, enforce_version=False)
    c = (df_clusters['eid'].nunique(), df_clusters['pid'].nunique(), df_clusters.shape[0], df_clusters['Beryl'].nunique())
    assert c == (459, 698, 65336, 266)
    print('Gray matter', *c)

    # min 5 units per region
    df_clusters = bwm_units(one=one, freeze='2023_12_bwm_release', rt_range=(0.08, 0.2), min_errors=3, min_qc=1, min_units_sessions=(5, 0), enforce_version=False)
    c = (df_clusters['eid'].nunique(), df_clusters['pid'].nunique(), df_clusters.shape[0], df_clusters['Beryl'].nunique())
    assert c == (455, 691, 63357, 245)
    print('min 5 units per regin', *c)

    # min 2 sessins per region
    df_clusters = bwm_units(one=one, freeze='2023_12_bwm_release', rt_range=(0.08, 0.2), min_errors=3, min_qc=1, min_units_sessions=(5, 2))
    c = (df_clusters['eid'].nunique(), df_clusters['pid'].nunique(), df_clusters.shape[0], df_clusters['Beryl'].nunique())
    assert c == (454, 690, 62_990, 210)
    print('min 2 sessions per region', *c)

