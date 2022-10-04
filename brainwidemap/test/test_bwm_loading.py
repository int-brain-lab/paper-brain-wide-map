from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

from iblutil.io.hashfile import md5

from brainwidemap import bwm_loading
import brainwidemap


def test_load_insertions():
    df_bwm = bwm_loading.bwm_query()
    hashes = pd.util.hash_pandas_object(df_bwm)
    assert(hashlib.sha1(pd.util.hash_array(hashes.values[np.newaxis, :]).tobytes()
                        ).hexdigest() == '56f88ec777f496bf8788973311e9610bcb21df0c')
    df_bwm, insertions = bwm_loading.bwm_query(return_details=True)
    assert df_bwm.shape[0] == 552
    assert len(insertions) == 552
    assert md5(Path(brainwidemap.__file__).parent.joinpath('fixtures', 'insertions_details.json')) == '8cd31ec7ff360558a888c2cc038d6b36'
