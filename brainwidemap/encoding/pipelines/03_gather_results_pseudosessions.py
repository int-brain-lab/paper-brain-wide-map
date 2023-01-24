# Standard library
from functools import cache

# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember

# Brainwidemap repo imports
from brainwidemap.encoding.utils import get_id, remap


def colrename(cname, suffix):
    return str(cname + 1) + "cov" + suffix


def remap(ids, source="Allen", dest="Beryl", output="acronym", br=BrainRegions()):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == "id":
        return br.id[br.mappings[dest][inds]]
    elif output == "acronym":
        return br.get(br.id[br.mappings[dest][inds]])["acronym"]


def get_id(acronym, brainregions=BrainRegions()):
    return brainregions.id[np.argwhere(brainregions.acronym == acronym)[0, 0]]


def get_name(acronym, brainregions=BrainRegions()):
    if acronym == "void":
        return acronym
    reg_idxs = np.argwhere(brainregions.acronym == acronym).flat
    return brainregions.name[reg_idxs[0]]


def label_cerebellum(acronym, brainregions=BrainRegions()):
    regid = brainregions.id[np.argwhere(brainregions.acronym == acronym).flat][0]
    ancestors = brainregions.ancestors(regid)
    if "Cerebellum" in ancestors.name or "Medulla" in ancestors.name:
        return True
    else:
        return False


def process_file(fitname):
    with open(fitname, "rb") as fo:
        tmpfile = pickle.load(fo)
    folds = []
    nulls = []
    for i in range(len(tmpfile["fitdata"]["scores"])):
        tmpdf = tmpfile["fitdata"]["deltas"][i]["test"]
        tmpdf["full_model"] = tmpfile["fitdata"]["scores"][i]["basescores"]["test"]
        tmpdf["eid"] = fitname.parts[-2]
        tmpdf["pid"] = fitname.parts[-1].split("_")[1]
        tmpdf["acronym"] = tmpfile["clu_regions"][tmpdf.index]
        tmpdf["qc_label"] = tmpfile["clu_df"]["label"][tmpdf.index]
        tmpdf["fold"] = i
        tmpdf["null"] = -1
        tmpdf.index.set_names(["clu_id"], inplace=True)
        folds.append(tmpdf.reset_index())
    for i in range(len(tmpfile["nullfits"])):
        for j in range(len(tmpfile["nullfits"][i]["scores"])):
            basescores = tmpfile["nullfits"][i]["scores"][j]["basescores"]["test"]
            nullscores = tmpfile["nullfits"][i]["scores"][j]["test"]
            nulldiffs = basescores.values.reshape(-1, 1) - nullscores
            nulldiffs = nulldiffs.droplevel("feature_iter")
            nulldiffs["full_model"] = basescores
            nulldiffs["eid"] = fitname.parts[-2]
            nulldiffs["pid"] = fitname.parts[-1].split("_")[1]
            nulldiffs["acronym"] = tmpfile["clu_regions"][nulldiffs.index]
            nulldiffs["qc_label"] = tmpfile["clu_df"]["label"][nulldiffs.index]
            nulldiffs["fold"] = j
            nulldiffs["null"] = i
            nulldiffs.index.set_names(["clu_id"], inplace=True)
            nulls.append(nulldiffs.reset_index())
    sess_master = pd.concat(folds)
    null_master = pd.concat(nulls)
    return sess_master, null_master


if __name__ == "__main__":
    # Standard library
    import os
    import pickle
    from pathlib import Path

    # Third party libraries
    from joblib import Parallel, delayed

    # Brainwidemap repo imports
    # Brainwide repo imports
    from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH

    FITDATE = "2022-12-23"  # Date on which fit was run
    PARALLEL = True

    kernels = [
        "stimonR",
        "stimonL",
        "correct",
        "incorrect",
        "fmoveR",
        "fmoveL",
        "pLeft",
        "pLeft_tr",
        "wheel",
        "full_model",
    ]

    parpath = Path(GLM_FIT_PATH).joinpath(f"{FITDATE}_glm_fit_pars.pkl")
    with open(parpath, "rb") as fo:
        params = pickle.load(fo)
    datapath = Path(GLM_CACHE).joinpath(params["dataset_fn"])
    with open(datapath, "rb") as fo:
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
                if os.path.isfile(filepath) and filepath.match(f"*{FITDATE}*"):
                    filenames.append(filepath)

    # Process files after fitting
    sessdfs = []
    nulldfs = []
    if PARALLEL:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_file)(fitname) for fitname in filenames
        )
        for sess, null in results:
            sessdfs.append(sess)
            nulldfs.append(null)
    else:
        for fitname in filenames:
            sess, null = process_file(fitname)
            sessdfs.append(sess)
            nulldfs.append(null)

    masterscores = pd.concat(sessdfs)
    nullscores = pd.concat(nulldfs)
    masterscores = pd.concat([masterscores, nullscores])
    meanmaster = (
        masterscores.set_index(["eid", "pid", "clu_id", "acronym", "qc_label", "null", "fold"])
        .groupby(["eid", "pid", "clu_id", "acronym", "qc_label", "null"])
        .agg({k: "mean" for k in kernels})
    )

    @cache
    def regmap(acr):
        ids = get_id(acr)
        return remap(ids, br=br)

    def dfrank(df):
        df.reset_index(inplace=True)
        return df[kernels].rank(axis=0, pct=True)[df.null == -1]

    br = BrainRegions()
    grpby = masterscores.groupby("acronym")
    meanmaster.reset_index(["acronym", "qc_label"], inplace=True)
    masterscores["region"] = [regmap(ac)[0] for ac in masterscores["acronym"]]
    meanmaster["region"] = [regmap(ac)[0] for ac in meanmaster["acronym"]]
    percentilemaster = meanmaster.groupby(["eid", "pid", "clu_id", "region"]).apply(dfrank)

    outdict = {
        "fit_params": params,
        "dataset": dataset,
        "fit_results": masterscores,
        "mean_fit_results": meanmaster,
        "percentiles": percentilemaster,
        "fit_files": filenames,
    }
    with open(Path(GLM_FIT_PATH).joinpath(f"{FITDATE}_glm_fit.pkl"), "wb") as fw:
        pickle.dump(outdict, fw)
