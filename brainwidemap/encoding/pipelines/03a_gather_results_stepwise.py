# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember


def compute_deltas(scores):
    outdf = pd.DataFrame(np.zeros_like(scores), index=scores.index, columns=scores.columns)
    for i in scores.columns:  # Change this for diff num covs
        if i >= 1:
            diff = scores[i] - scores[i - 1]
        else:
            diff = scores[i]
        outdf[i] = diff
    return outdf


def colrename(cname, suffix):
    return str(cname + 1) + 'cov' + suffix


def remap(ids, source='Allen', dest='Beryl', output='acronym', br=BrainRegions()):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def get_id(acronym, brainregions=BrainRegions()):
    return brainregions.id[np.argwhere(brainregions.acronym == acronym)[0, 0]]


def get_name(acronym, brainregions=BrainRegions()):
    if acronym == 'void':
        return acronym
    reg_idxs = np.argwhere(brainregions.acronym == acronym).flat
    return brainregions.name[reg_idxs[0]]


def label_cerebellum(acronym, brainregions=BrainRegions()):
    regid = brainregions.id[np.argwhere(brainregions.acronym == acronym).flat][0]
    ancestors = brainregions.ancestors(regid)
    if 'Cerebellum' in ancestors.name or 'Medulla' in ancestors.name:
        return True
    else:
        return False


if __name__ == "__main__":
    # Standard library
    import os
    import pickle
    from pathlib import Path

    # Brainwide repo imports
    from brainwide.params import GLM_CACHE, GLM_FIT_PATH

    currdate = "2022-02-14"  # Date on which fit was run
    n_cov = 8  # Modify if you change the model!
    parpath = Path(GLM_FIT_PATH).joinpath(f'{currdate}_glm_fit_pars.pkl')
    with open(parpath, 'rb') as fo:
        params = pickle.load(fo)
    datapath = Path(GLM_CACHE).joinpath(params['dataset_fn'])
    with open(datapath, 'rb') as fo:
        dataset = pickle.load(fo)

    filenames = []
    for subj in os.listdir(Path(GLM_FIT_PATH)):
        subjdir = Path(GLM_FIT_PATH).joinpath(subj)
        if not os.path.isdir(subjdir):
            continue
        for sess in os.listdir(subjdir):
            sessdir = subjdir.joinpath(sess)
            for file in os.listdir(sessdir):
                filepath = sessdir.joinpath(file)
                if os.path.isfile(filepath) and filepath.match(f'*{currdate}*'):
                    filenames.append(filepath)

    # Process files after fitting
    sessdfs = []
    for fitname in filenames:
        with open(fitname, 'rb') as fo:
            tmpfile = pickle.load(fo)
        folds = []
        for i in range(len(tmpfile['scores'])):
            tmp_sc = tmpfile['scores'][i].rename(columns=lambda c: colrename(c, '_score'))
            tmp_seq = tmpfile['sequences'][i].rename(columns=lambda c: colrename(c, '_name'))
            tmp_diff = compute_deltas(tmpfile['scores'][i])
            tmp_diff.rename(columns=lambda c: colrename(c, '_diff'), inplace=True)
            tmpdf = tmp_sc.join(tmp_seq).join(tmp_diff)
            tmpdf['eid'] = fitname.parts[-2]
            tmpdf['acronym'] = tmpfile['clu_regions'][tmpdf.index]
            tmpdf['qc_label'] = tmpfile['clu_qc']['label'][tmpdf.index]
            tmpdf['fold'] = i
            tmpdf.index.set_names(['clu_id'], inplace=True)
            folds.append(tmpdf.reset_index())
        sess_master = pd.concat(folds)
        sessdfs.append(sess_master)
    masterscores = pd.concat(sessdfs)

    for i in range(1, n_cov + 1):  # Change this for diff num covs
        if i >= 2:
            diff = masterscores[str(i) + 'cov_score'] - masterscores[str(i - 1) + 'cov_score']
        else:
            diff = masterscores[str(i) + 'cov_score']
    masterscores[str(i) + 'cov_diff'] = diff

    br = BrainRegions()
    grpby = masterscores.groupby('acronym')
    masterscores['reg_id'] = grpby.acronym.transform(lambda g: get_id(g.unique()[0], br))
    masterscores['beryl_acr'] = grpby.reg_id.transform(lambda g: remap(g, br=br))
    masterscores['cerebellum'] = grpby.acronym.transform(
        lambda g: label_cerebellum(g.unique()[0], br))
    masterscores['region'] = masterscores['beryl_acr']
    masterscores['name'] = grpby.region.transform(lambda g: get_name(g.unique()[0], br))

    masterscores.set_index(['eid', 'acronym', 'clu_id', 'fold'], inplace=True)

    outdict = {
        'fit_params': params,
        'dataset': dataset,
        'fit_results': masterscores,
        'fit_files': filenames,
    }
    with open(Path(GLM_FIT_PATH).joinpath(f'{currdate}_glm_fit.pkl'), 'wb') as fw:
        pickle.dump(outdict, fw)
