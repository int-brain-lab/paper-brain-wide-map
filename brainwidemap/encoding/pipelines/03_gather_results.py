# Standard library
from functools import cache

# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember


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


if __name__ == "__main__":
    # Standard library
    import os
    import pickle
    from pathlib import Path

    # Brainwidemap repo imports
    # Brainwide repo imports
    from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH

    currdate = "2024-09-09"  # Date on which fit was run
    n_cov = 9  # Modify if you change the model!
    parpath = Path(GLM_FIT_PATH).joinpath(f"{currdate}_glm_fit_pars.pkl")
    early_split = False
    with open(parpath, "rb") as fo:
        params = pickle.load(fo)
    if "rt_thresh" in params:
        early_split = True
        n_cov += 2
    datapath = Path(GLM_CACHE).joinpath(params["dataset_fn"])
    with open(datapath, "rb") as fo:
        dataset = pickle.load(fo)
    subject_names = dataset["dataset_filenames"]["subject"].unique()

    filenames = []
    for subj in os.listdir(Path(GLM_FIT_PATH)):
        if subj not in subject_names:
            continue
        subjdir = Path(GLM_FIT_PATH).joinpath(subj)
        if not os.path.isdir(subjdir):
            continue
        for sess in os.listdir(subjdir):
            sessdir = subjdir.joinpath(sess)
            for file in os.listdir(sessdir):
                filepath = sessdir.joinpath(file)
                if os.path.isfile(filepath) and filepath.match(f"*{currdate}*"):
                    filenames.append(filepath)

    # Process files after fitting
    sessdfs = []
    for fitname in filenames:
        with open(fitname, "rb") as fo:
            tmpfile = pickle.load(fo)
        folds = []
        for i in range(len(tmpfile["scores"])):
            tmpdf = tmpfile["deltas"][i]["test"]
            tmpdf.index.name = "clu_id"
            tmpdf["full_model"] = tmpfile["scores"][i]["basescores"]["test"]
            tmpdf["eid"] = fitname.parts[-2]
            tmpdf["pid"] = fitname.parts[-1].split("_")[1]
            tmpdf["acronym"] = tmpfile["clu_regions"]
            tmpdf["qc_label"] = tmpfile["clu_df"]["label"]
            tmpdf["fold"] = i
            tmpdf.index = tmpfile["clu_df"].iloc[tmpdf.index].cluster_id
            tmpdf.index.set_names(["clu_id"], inplace=True)
            folds.append(tmpdf.reset_index())
        sess_master = pd.concat(folds)
        sessdfs.append(sess_master)
    masterscores = pd.concat(sessdfs)
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
    if early_split:
        kernels.insert(6, "fmoveL_early")
        kernels.insert(6, "fmoveR_early")

    meanmaster = (
        masterscores.set_index(["eid", "pid", "clu_id", "acronym", "qc_label", "fold"])
        .groupby(["eid", "pid", "clu_id", "acronym", "qc_label"])
        .agg({k: "mean" for k in kernels})
    )

    @cache
    def regmap(acr):
        ids = get_id(acr)
        return remap(ids, br=br)

    br = BrainRegions()
    grpby = masterscores.groupby("acronym")
    meanmaster.reset_index(["acronym", "qc_label"], inplace=True)
    masterscores["region"] = [regmap(ac)[0] for ac in masterscores["acronym"]]
    meanmaster["region"] = [regmap(ac)[0] for ac in meanmaster["acronym"]]

    outdict = {
        "fit_params": params,
        "dataset": dataset,
        "fit_results": masterscores,
        "mean_fit_results": meanmaster,
        "fit_files": filenames,
    }
    with open(Path(GLM_FIT_PATH).joinpath(f"{currdate}_glm_fit.pkl"), "wb") as fw:
        pickle.dump(outdict, fw)
