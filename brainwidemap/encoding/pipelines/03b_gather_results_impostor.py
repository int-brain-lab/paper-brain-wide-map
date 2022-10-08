# Standard library
from collections import defaultdict
from functools import cache

# Third party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# IBL libraries
from ibllib.atlas import BrainRegions

# Brainwide repo imports
from brainwide.utils import get_id, remap


def generate_da_dict(
        filename,
        n_impostors,
        n_folds,
        region_mapper=lambda x: x,):
    datafile = pd.read_pickle(filename)
    eid = filename.parts[-2]  # Ugly hack because I forgot to keep eid value in files
    dfs = process_kernels(datafile, eid, n_impostors, n_folds, region_mapper)
    return dfs


def process_kernels(datafile, eid, n_impostors, n_folds, reg_map, get_kweights=False):
    dfs = defaultdict(list)
    for kernel in datafile['fitdata']['weights'][0].keys():
        for i in range(-1, n_impostors):
            for j in range(n_folds):
                if i < 0:  # this references the actual data fit
                    df = datafile['fitdata']['weights'][j][kernel].copy()
                    ref_idx = df.index.copy()
                else:
                    df = datafile['nullfits'][i]['weights'][j][kernel].copy().reindex(ref_idx)
                mi = make_multiindex(datafile, eid, reg_map, df, j, i)
                df.index = mi
                dfs[kernel].append(df)
    dfs_cat = {k: pd.concat(dfs[k]).sort_index() for k in dfs}
    return dfs_cat


def make_multiindex(datafile, eid, reg_map, df, fold, impostor):
    mi = pd.MultiIndex.from_arrays(
        [
            np.array([eid] * len(df)),
            df.index,
            np.array([reg_map(id) for id in datafile['clu_regions'][df.index]]).flatten(),
            np.ones_like(df.index) * impostor,
            np.ones_like(df.index) * fold,
        ],
        names=['eid', 'clu_id', 'region', 'nullrun', 'fold'],
    )
    return mi


def extract_kweights(df: pd.DataFrame, bases=None):
    def lstsq(row):
        try:
            return np.linalg.lstsq(bases, row)[0]
        except np.linalg.LinAlgError:
            return np.nan * np.ones(bases.shape[1])

    if len(df.columns) == 1:
        return df
    else:
        newdf = df.apply(lstsq, axis=1, result_type='expand')
        newdf.columns = np.arange(bases.shape[1])
        return newdf


if __name__ == '__main__':
    # Standard library
    import os
    import pickle
    from pathlib import Path

    # Brainwide repo imports
    from brainwide.params import GLM_CACHE, GLM_FIT_PATH

    FITDATE = '2022-02-24'  # Date on which fit was run

    parpath = Path(GLM_FIT_PATH).joinpath(f'{FITDATE}_glm_fit_pars.pkl')
    with open(parpath, 'rb') as fo:
        params = pickle.load(fo)
    datapath = Path(GLM_CACHE).joinpath(params['dataset_fn'])
    with open(datapath, 'rb') as fo:
        dataset = pickle.load(fo)

    # Make a directory to store results from this fit run
    fitfolder = Path(GLM_FIT_PATH).joinpath('merged_results').joinpath(f'{FITDATE}_impostor_run')
    try:
        os.mkdir(fitfolder)
    except FileExistsError:
        pass

    n_folds = params['n_folds'] if 'n_folds' in params else 5
    n_impostors = params['n_impostors']

    filenames = []
    for subj in os.listdir(Path(GLM_FIT_PATH)):
        if subj == 'merged_results':
            continue
        subjdir = Path(GLM_FIT_PATH).joinpath(subj)
        if not os.path.isdir(subjdir):
            continue
        for sess in os.listdir(subjdir):
            sessdir = subjdir.joinpath(sess)
            for file in os.listdir(sessdir):
                filepath = sessdir.joinpath(file)
                if os.path.isfile(filepath) and filepath.match(f'*{FITDATE}*impostor*'):
                    filenames.append(filepath)

    br = BrainRegions()

    @cache
    def regmap(acr):
        ids = get_id(acr)
        return remap(ids, br=br)

    fdata = defaultdict(list)
    for filename in tqdm(filenames):
        dfs = generate_da_dict(filename, n_impostors, n_folds, regmap)
        for k in dfs:
            fdata[k].append(dfs[k])

    for k in list(fdata.keys()):
        fdata[k] = pd.concat(fdata[k])
        if isinstance(fdata[k], pd.DataFrame):
            fdata[k].columns = fdata[k].columns.astype(str)
        else:
            fdata[k] = fdata[k].to_frame()
        fdata[k].to_parquet(fitfolder.joinpath(f'{k}_data.parquet'))
        del fdata[k]

    with open(fitfolder.joinpath('fit_params.pkl'), 'wb') as fw:
        pickle.dump(params, fw)
    with open(fitfolder.joinpath('fit_dataset.pkl'), 'wb') as fw:
        pickle.dump(dataset, fw)
