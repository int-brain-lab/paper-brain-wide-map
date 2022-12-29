# Standard library
import argparse
import pickle
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold

# IBL libraries
from brainbox.task.closed_loop import generate_pseudo_blocks

# Brainwidemap repo imports
from brainwidemap.encoding.cluster_worker import get_cached_regressors
from brainwidemap.encoding.params import GLM_CACHE

parser = argparse.ArgumentParser(description="Run single unit regression on only ITI P(Left)")
parser.add_argument("--cachedate", type=str, default="2022-12-22")
parser.add_argument("--pseudo", type=int, default=100)
parser.add_argument("--contiguous", action="store_true")
parser.add_argument("--fileidx", type=int, default=-1)
args = parser.parse_args()

dsfn = Path(GLM_CACHE + f"{args.cachedate}_dataset_metadata.pkl")
dataset = pd.read_pickle(dsfn)["dataset_filenames"]
alphas = np.logspace(-2, 1.5, 50)
estimator = GridSearchCV(lm.Ridge(), {"alpha": alphas})
parpool = Parallel(n_jobs=-1, verbose=10)


def fit_target(binned, target, clu_idx):
    scores = np.zeros(5)
    for i, fold in enumerate(KFold(n_splits=5, shuffle=not args.contiguous).split(target)):
        train, test = fold
        fit = estimator.fit(target[train].reshape(-1, 1), binned[train, clu_idx])
        scores[i] = fit.score(target[test].reshape(-1, 1), binned[test, clu_idx])
    return scores


allscores = []
allmeanscores = []
allpct = []

for i, row in dataset.iterrows():
    if args.fileidx != -1 and i != args.fileidx:
        continue
    subject, eid, pid, _, reg_file = row
    print(subject, eid, pid)
    stdf, sspkt, sspkclu, sclureg, scluqc = get_cached_regressors(reg_file)
    windows = pd.concat(((stdf["stimOn_times"] - 0.4), (stdf["stimOn_times"] - 0.1)), axis=1)
    windows.columns = ["start", "end"]

    # Add extra clu since we're binning them and not counting
    clu_ids = np.pad(np.unique(sspkclu), (0, 1), constant_values=sspkclu.max() + 1)
    binned, tbins, clu_ids = np.histogram2d(
        sspkclu, sspkt, bins=(windows.values.flatten(), clu_ids)
    )
    binned = binned[::2]
    target = stdf["probabilityLeft"].values
    basescores = parpool(
        delayed(fit_target)(binned, target, clu_idx) for clu_idx in range(binned.shape[1])
    )
    basescores = pd.DataFrame(basescores, index=clu_ids[:-1], columns=range(5))
    basescores["null"] = -1
    basescores.set_index("null", append=True, inplace=True)
    basescores.index.names = ["clu_id", "null"]
    null_targets = [
        generate_pseudo_blocks(stdf.index.max() + 1)[stdf.index] for _ in range(args.pseudo)
    ]
    nullfits = parpool(
        delayed(fit_target)(binned, nt, clu_idx)
        for nt in null_targets
        for clu_idx in range(binned.shape[1])
    )
    indices = [(clu, null) for null in range(len(null_targets)) for clu in clu_ids[:-1]]
    nullscores = pd.DataFrame(nullfits, index=pd.MultiIndex.from_tuples(indices), columns=range(5))
    nullscores.index.names = ["null", "clu_id"]
    scores = pd.concat((basescores, nullscores))

    meanscores = scores.mean(axis=1)
    percentiles = meanscores.groupby("clu_id").apply(lambda df: df.rank(pct=True).loc[:, -1])
    percentiles = percentiles.droplevel(0)

    scores["eid"] = eid
    scores["pid"] = pid
    meanscores["eid"] = eid
    meanscores["pid"] = pid
    percentiles["eid"] = eid
    percentiles["pid"] = pid

    allscores.append(scores)
    allmeanscores.append(meanscores)
    allpct.append(percentiles)

outdict = {
    "scores": pd.concat(allscores),
    "meanscores": pd.concat(allmeanscores),
    "percentiles": pd.concat(allpct),
}
idxstr = "" if args.fileidx == -1 else f"_idx{args.fileidx}"
filename = Path(GLM_CACHE + f"{args.cachedate}_single_unit_iti_pleft_regression{idxstr}.pkl")
with open(filename, "wb") as f:
    pickle.dump(outdict, f)
