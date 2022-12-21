# Author: Mayo Faulkner

# Run this on SDSC
import numpy as np
from ibllib.io.video import get_video_meta
from pathlib import Path
from one.api import ONE
import pandas as pd
from brainwidemap import bwm_query

df = bwm_query()
eids = df.eid.unique()
one = ONE()

root_path = Path("/mnt/ibl")
fi_path = Path("https://ibl.flatironinstitute.org/")

all_info = []
errored_eids = []

for i, eid in enumerate(eids):

    if np.mod(i, 10) == 0:
        print(i)
    for lab in ['left', 'right', 'body']:
        try:

            dset_vid = one.alyx.rest('datasets', 'list', session=eid, name=f'_iblrig_{lab}Camera.raw.mp4', no_cache=True)
            dset_time = one.alyx.rest('datasets', 'list', session=eid, name=f'_ibl_{lab}Camera.times.npy', no_cache=True)

            video_exists = len(dset_vid) == 1
            time_exists = len(dset_time) == 1
            mismatch = None
            n_mismatch = None
            duration = None

            if video_exists and time_exists:

                url_vid = next(fr['data_url'] for fr in dset_vid[0]['file_records'] if 'flatiron' in fr['data_repository'])
                d_vid_path = root_path.joinpath(Path(url_vid).relative_to(fi_path))

                url_time = next(fr['data_url'] for fr in dset_time[0]['file_records'] if 'flatiron' in fr['data_repository'])
                d_time_path = root_path.joinpath(Path(url_time).relative_to(fi_path))

                meta = get_video_meta(d_vid_path)
                vid_size = meta['length']
                ts_size = np.load(d_time_path).size
                duration = meta['duration'].seconds

                if vid_size != ts_size:
                    mismatch = True
                    n_mismatch = ts_size - vid_size
                else:
                    mismatch = False
                    n_mismatch = 0
            elif video_exists:
                url_vid = next(fr['data_url'] for fr in dset_vid[0]['file_records'] if 'flatiron' in fr['data_repository'])
                d_vid_path = root_path.joinpath(Path(url_vid).relative_to(fi_path))
                duration = get_video_meta(d_vid_path)['duration'].seconds

            info = {'eid': eid,
                    'label': lab,
                    'video': video_exists,
                    'times': time_exists,
                    'mismatch': mismatch,
                    'n_mismatch': n_mismatch,
                    'duration': duration}

            all_info.append(info)

        except Exception as err:
            errored_eids.append(f'{eid}_{lab}')

    all_df = pd.DataFrame.from_dict(all_info)
    all_df.to_csv('/mnt/ibl/resources/video_status_sdsc.csv')
