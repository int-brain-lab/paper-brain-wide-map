# Standard library
from functools import cache
from argparse import ArgumentParser

# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
from ibllib.atlas import BrainRegions


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
        default="2023-04-09",
        help="Date on which fit was run",
    )
    parser.add_argument(
        "--n_cov",
        type=int,
        default=9,
        help="Number of covariates in model",
    )
    args = parser.parse_args()
    fitdate = args.fitdate
    n_cov = args.n_cov
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

    # Process files after fitting
    sessdfs = []
    for fitname in filenames:
        with open(fitname, "rb") as fo:
            tmpfile = pickle.load(fo)
        folds = []
        for i in range(len(tmpfile["scores"])):
            tmpdf = tmpfile["deltas"][i]["test"]
            tmpdf["full_model"] = tmpfile["scores"][i]["basescores"]["test"]
            tmpdf["eid"] = fitname.parts[-2]
            tmpdf["pid"] = fitname.parts[-1].split("_")[1]
            tmpdf["acronym"] = tmpfile["clu_regions"][tmpdf.index]
            tmpdf["qc_label"] = tmpfile["clu_df"]["label"][tmpdf.index]
            tmpdf["fold"] = i
            tmpdf.index.set_names(["clu_id"], inplace=True)
            folds.append(tmpdf.reset_index())
        sess_master = pd.concat(folds)
        sessdfs.append(sess_master)
    masterscores = pd.concat(sessdfs)
    # Take the average score across the different folds of cross-validation for each unit for
    # each of the model regressors
    meanmaster = (
        masterscores.set_index(["eid", "pid", "clu_id", "acronym", "qc_label", "fold"])
        .groupby(["eid", "pid", "clu_id", "acronym", "qc_label"])
        .agg(
            {
                k: "mean"
                for k in [
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
            }
        )
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

    outdict = {
        "fit_params": params,
        "dataset": dataset,
        "fit_results": masterscores,
        "mean_fit_results": meanmaster,
        "fit_files": filenames,
    }
    with open(Path(GLM_FIT_PATH).joinpath(f"{fitdate}_glm_fit.pkl"), "wb") as fw:
        pickle.dump(outdict, fw)
