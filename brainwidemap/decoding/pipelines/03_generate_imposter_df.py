from pathlib import Path
import pandas as pd
from one.api import ONE
import brainbox.io.one as bbone
from brainbox.task.closed_loop import generate_pseudo_session

FAKE_IMPOSTER_SESSION = False  # only here for debugging
GENERATE_FROM_EPHYS = False  # the number of ephys session template is too small
# DECODING_PATH = Path("/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi/decoding/")
DECODING_PATH = Path("/home/users/f/findling/scratch")

one = ONE()

if GENERATE_FROM_EPHYS:
    insdf = pd.read_parquet(DECODING_PATH.joinpath('insertions.pqt'))
    insdf = insdf[insdf.spike_sorting != '']
    eids = insdf['eid'].unique()
else:
    eids = one.search(project='ibl_neuropixel_brainwide_01',
                      task_protocol='biasedChoiceWorld')  # no template, no neural activity

columns = ['probabilityLeft', 'contrastRight', 'feedbackType', 'choice', 'contrastLeft', 'eid', 'template_sess']

all_trialsdf = []
for i, u in enumerate(eids):
    try:
        det = one.get_details(u, full=True)
        print(i)
        # mice on the rig and more than 400 trials and better then 90% on highest contrasts trials (BWM criteria)
        if 'ephys' in det['json']['PYBPOD_BOARD']:
            trialsdf = bbone.load_trials_df(u, one=one)
            if ((trialsdf.index.size > 400) and
                ((trialsdf[(trialsdf.contrastLeft == 1) |
                           (trialsdf.contrastRight == 1)].feedbackType == 1).mean() > 0.9) and
                ((trialsdf.probabilityLeft == 0.5).sum() == 90) and (trialsdf.probabilityLeft.values[0] == 0.5)):
                session_id = i # if not GENERATE_FROM_EPHYS else det['json']['SESSION_ORDER'][det['json']['SESSION_IDX']]
                if FAKE_IMPOSTER_SESSION:
                    trialsdf = generate_pseudo_session(trialsdf)
                trialsdf['eid'] = u
                trialsdf['trial_id'] = trialsdf.index
                trialsdf['template_sess'] = session_id
                all_trialsdf.append(trialsdf)
    except Exception as e:
        print(e)

all_trialsdf = pd.concat(all_trialsdf)

# save imposter sessions
all_trialsdf[columns].to_parquet(DECODING_PATH.joinpath('imposterSessions_beforeRecordings.pqt'))

'''
# ?todo add eid template https://github.com/int-brain-lab/iblenv/issues/117
# ?todo change this with good performing behavioral sessions? not ephys sessions
#  get eids of behavioral sessions
one = ONE()
# task_protocol='ephysChoiceWorld' for template sessions with neural activity
eids_behav = one.search(project='ibl_neuropixel_brainwide_01',
                        task_protocol='biasedChoiceWorld',
                        )  # no template, no neural activity
for u in eids:
    det = one.get_details(u, full=True)
    if 'ephys' in det['json']['PYBPOD_BOARD']: # mice are on the ephys rig but no neural recordings
        # do stuff
        det = one.get_details(u, full=True)
        eid = det['json']['SESSION_ORDER'][det['json']['SESSION_IDX']]

# original code used when I did now know how to get the session id
pLeft_MIN_BEHAV_TRIAS = np.vstack([all_trialsdf[(all_trialsdf.trial_id < MIN_BEHAV_TRIAS) & (all_trialsdf.eid == u)]
                                   .probabilityLeft.values
                                   for u in all_trialsdf.eid.unique()])
pLeft_MIN_BEHAV_TRIAS_uniq = np.unique(pLeft_MIN_BEHAV_TRIAS, axis=0)

if pLeft_MIN_BEHAV_TRIAS_uniq.shape[0] != 12:
    raise ValueError('these is most likely a bug in this pipeline')

template_sess = np.argmax(np.all(pLeft_MIN_BEHAV_TRIAS[None] == pLeft_MIN_BEHAV_TRIAS_uniq[:, None], axis=-1), axis=0)
all_trialsdf['template_sess'] = template_sess[np.argmax(all_trialsdf.eid.values[:, None]
                                                        == all_trialsdf.eid.unique()[None], axis=-1)]
'''
