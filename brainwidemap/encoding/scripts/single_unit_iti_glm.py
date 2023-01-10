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
import neurencoding.design_matrix as dm
import neurencoding.linear as nl
from brainbox.task.closed_loop import generate_pseudo_blocks

# Brainwidemap repo imports
from brainwidemap.encoding.cluster_worker import get_cached_regressors
from brainwidemap.encoding.params import GLM_CACHE


def gen_design(stdf, binwidth=0.3):
    newdf = stdf[["stimOn_times"]]
    newdf["trial_start"] = newdf["stimOn_times"] - 0.4
    newdf["trial_end"] = newdf["stimOn_times"] - 0.1
    newdf["prior_last"] = pd.Series(np.roll(stdf["probabilityLeft"], 1), index=stdf.index)
    vartypes = {
        "trial_start": "timing",
        "trial_end": "timing",
        "stimOn_times": "timing",
        "probabilityLeft": "value",
        "prior_last": "value",
    }

    def stepfunc_prestim(row):
        stepvec = np.ones(design.binf(row.duration)) * row.prior_last
        return stepvec

    design = dm.DesignMatrix(newdf, vartypes, binwidth=binwidth)
    design.add_covariate_raw("pLeft", stepfunc_prestim, desc="Step function on prior")
    design.compile_design_matrix()
    return design


def fit_target(sspkt, sspkclu, stdf, binwidth):
    scores = []
    estimator = GridSearchCV(lm.Ridge(), {"alpha": np.logspace(-2, 1.5, 50)})
    design = gen_design(stdf, binwidth=binwidth)
    nglm = nl.LinearGLM(
        design, sspkt, sspkclu, binwidth=binwidth, estimator=estimator, mintrials=0
    )
    for i, fold in enumerate(KFold(n_splits=5, shuffle=not args.contiguous).split(stdf.index)):
        train, test = fold
        trdx = stdf.index[train]
        tedx = stdf.index[test]
        nglm.fit(train_idx=trdx)
        scores.append(nglm.score(testinds=tedx).to_frame().assign(fold=i))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single unit regression on only ITI P(Left)")
    parser.add_argument("--cachedate", type=str, default="2022-12-22")
    parser.add_argument("--pseudo", type=int, default=100)
    parser.add_argument("--binwidth", type=float, default=0.3)
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--fileidx", type=int, default=-1)
    parser.add_argument("--nthreads", type=int, default=-1)
    args = parser.parse_args()

    dsfn = Path(GLM_CACHE + f"{args.cachedate}_dataset_metadata.pkl")
    dataset = pd.read_pickle(dsfn)["dataset_filenames"]
    parpool = Parallel(n_jobs=args.nthreads, verbose=10)

    allscores = []
    allmeanscores = []
    allpct = []

    for i, row in dataset.iterrows():
        if args.fileidx != -1 and i != args.fileidx:
            continue
        subject, eid, pid, _, reg_file = row
        print(subject, eid, pid)
        relpath = "/".join(Path(reg_file).parts[-3:])
        relfile = Path(GLM_CACHE + relpath)
        stdf, sspkt, sspkclu, sclureg, scluqc = get_cached_regressors(relfile)

        # Add extra clu since we're binning them and not counting
        basescores = fit_target(sspkt, sspkclu, stdf, args.binwidth)
        basescores = pd.concat(basescores)
        basescores["null"] = -1
        null_targets = [
            generate_pseudo_blocks(stdf.index.max() + 1)[stdf.index] for _ in range(args.pseudo)
        ]
        null_dfs = [stdf.copy().assign(probabilityLeft=t) for t in null_targets]
        nullfits = parpool(
            delayed(fit_target)(sspkt, sspkclu, nt, args.binwidth)
            for nt in null_dfs
        )
        nulldfs = [pd.concat(scores).assign(null=i) for i, scores in enumerate(nullfits)]
        nullscores = pd.concat(nulldfs)
        scores = pd.concat((basescores, nullscores))
        scores.set_index(["null", "fold"], append=True, inplace=True)
        scores.index.names = ["clu_id", "null", "fold"]

        meanscores = scores.groupby(["clu_id", "null"]).mean()
        percentiles = meanscores.groupby("clu_id").apply(lambda df: df.rank(pct=True).loc[:, -1, :])
        percentiles = percentiles.droplevel(0).to_frame()
        meanscores = meanscores.to_frame()

        scores["eid"] = eid
        scores["pid"] = pid
        meanscores["eid"] = eid
        meanscores["pid"] = pid
        percentiles["eid"] = eid
        percentiles["pid"] = pid

        scores = scores.reset_index().set_index(["eid", "pid", "clu_id", "null"]).sort_index()
        meanscores = (
            meanscores.reset_index().set_index(["eid", "pid", "clu_id", "null"]).sort_index()
        )
        percentiles = percentiles.reset_index().set_index(["eid", "pid", "clu_id"]).sort_index()

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
