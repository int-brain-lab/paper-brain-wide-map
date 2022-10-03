from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

from iblutil.io.hashfile import md5

from brainwidemap import bwm_loading
import brainwidemap


def test_data_freeze():
    df_bwm = bwm_loading.bwm_query()
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '56f88ec777f496bf8788973311e9610bcb21df0c')
    assert df_bwm.shape[0] == 552
