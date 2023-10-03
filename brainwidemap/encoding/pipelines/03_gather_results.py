# Standard library
from functools import cache
from argparse import ArgumentParser

# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
from ibllib.atlas import BrainRegions


def covarmap(params, covars):
    names = []
    for cov in covars:
        match cov:
            case "stimonR" | "stimonL":
                covshape = params["bases"]["stim"].shape[1]
            case "fmoveR" | "fmoveL":
                covshape = params["bases"]["fmove"].shape[1]
            case "correct" | "incorrect":
                covshape = params["bases"]["feedback"].shape[1]
            case "pLeft" | "pLeft_tr":
                covshape = 1
            case "wheel":
                covshape = params["bases"]["wheel"].shape[1]
        if covshape == 1:
            names.append(cov)
        else:
            names.extend([f"{cov}_{i}" for i in range(covshape)])
    return names


if __name__ == "__main__":
    # Standard library
    import os
    import pickle
    from pathlib import Path

    # Brainwidemap repo imports
    # Brainwide repo imports
    from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH

    parser = ArgumentParser(
        description="Gather results from GLM fitting on a given date with given N covariates."
    )
    parser.add_argument(
        "--fitdate",
        type=str,
        default="2023-10-02",
        help="Date on which fit was run",
    )
    args = parser.parse_args()
    fitdate = args.fitdate
    parpath = Path(GLM_FIT_PATH).joinpath(f"{fitdate}_glm_fit_pars.pkl")
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
                if os.path.isfile(filepath) and filepath.match(f"*{fitdate}*"):
                    filenames.append(filepath)

    covnames = [
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

    # Process files after fitting
    sessdfs = []
    sessweights = []
    weightnames = {i: n for i, n in enumerate(covarmap(params, covnames))}

    for fitname in filenames:
        with open(fitname, "rb") as fo:
            tmpfile = pickle.load(fo)
        weights = pd.DataFrame.from_dict(
            tmpfile["fullfitpars"]["coefficients"].to_dict(), orient="index"
        )
        scores_folds = []
        for i in range(len(tmpfile["scores"])):
            tmpdf = tmpfile["deltas"][i]["test"]
            tmpdf["full_model"] = tmpfile["scores"][i]["basescores"]["test"]
            tmpdf["eid"] = fitname.parts[-2]
            tmpdf["pid"] = fitname.parts[-1].split("_")[1]
            tmpdf["acronym"] = tmpfile["clu_regions"][tmpdf.index]
            tmpdf["qc_label"] = tmpfile["clu_df"]["label"][tmpdf.index]
            tmpdf["fold"] = i
            tmpdf.index.set_names(["clu_id"], inplace=True)
            scores_folds.append(tmpdf.reset_index())
        weights["full_model"] = tmpfile["scores"][i]["basescores"]["test"]
        weights["eid"] = fitname.parts[-2]
        weights["pid"] = fitname.parts[-1].split("_")[1]
        weights["acronym"] = tmpfile["clu_regions"][weights.index]
        weights["qc_label"] = tmpfile["clu_df"]["label"][weights.index]
        weights.rename(columns=weightnames, inplace=True)

        sess_master = pd.concat(scores_folds)
        sessdfs.append(sess_master)
        sessweights.append(weights)
    masterscores = pd.concat(sessdfs)
    masterweights = pd.concat(sessweights)
    masterweights.index.name = "clu_id"
    # Take the average score across the different folds of cross-validation for each unit for
    # each of the model regressors
    meanmaster = (
        masterscores.set_index(["eid", "pid", "clu_id", "acronym", "qc_label", "fold"])
        .groupby(["eid", "pid", "clu_id", "acronym", "qc_label"])
        .agg({k: "mean" for k in covnames})
    )

    br = BrainRegions()

    @cache
    def regmap(acr):
        return br.acronym2acronym(acr, mapping="Beryl")

    br = BrainRegions()
    # Remap the existing acronyms, which use the Allen ontology, into the Beryl ontology
    # Note that the groupby operation is to save time on computation so we don't need to
    # recompute the region mapping for each unit, but rather each Allen acronym.
    grpby = masterscores.groupby("acronym")
    meanmaster.reset_index(["acronym", "qc_label"], inplace=True)
    masterscores["region"] = [regmap(ac)[0] for ac in masterscores["acronym"]]
    meanmaster["region"] = [regmap(ac)[0] for ac in meanmaster["acronym"]]
    masterweights["region"] = [regmap(ac)[0] for ac in masterweights["acronym"]]

    outdict = {
        "fit_params": params,
        "dataset": dataset,
        "fit_results": masterscores,
        "mean_fit_results": meanmaster,
        "fit_weights": masterweights.reset_index().set_index(
            ["eid", "pid", "clu_id", "acronym", "qc_label"]
        ),
        "fit_files": filenames,
    }
    with open(Path(GLM_FIT_PATH).joinpath(f"{fitdate}_glm_fit.pkl"), "wb") as fw:
        pickle.dump(outdict, fw)
