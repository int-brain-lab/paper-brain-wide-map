# Third party libraries
from joblib import parallel_backend
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

# IBL libraries
from brainbox.task.closed_loop import generate_pseudo_blocks
import neurencoding.utils as mut

# Brainwide repo imports
from .design import generate_design, sample_impostor


def fit(design, spk_t, spk_clu, binwidth, model, estimator, n_folds=5, contiguous=False, **kwargs):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=not contiguous)
    scores, weights, intercepts, alphas, splits = [], [], [], [], []
    for test, train in splitter.split(trials_idx):
        nglm.fit(train_idx=trials_idx[train], printcond=False)
        if isinstance(estimator, GridSearchCV):
            alphas.append(estimator.best_params_["alpha"])
        elif isinstance(estimator, RegressorMixin):
            alphas.append(estimator.get_params()["alpha"])
        else:
            raise TypeError("Estimator must be a sklearn linear regression instance")
        intercepts.append(nglm.intercepts)
        weights.append(nglm.combine_weights(peaksonly=True))
        scores.append(nglm.score(testinds=trials_idx[test]))
        splits.append({"test": test, "train": train})
    outdict = {
        "scores": scores,
        "weights": weights,
        "intercepts": intercepts,
        "alphas": alphas,
        "splits": splits,
    }
    return outdict


def fit_stepwise(
    design,
    spk_t,
    spk_clu,
    binwidth,
    model,
    estimator,
    n_folds=5,
    contiguous=False,
    seqsel_kwargs={},
    seqselfit_kwargs={},
    **kwargs
):
    trials_idx = design.trialsdf.index
    nglm = model(design, spk_t, spk_clu, binwidth=binwidth, estimator=estimator)
    splitter = KFold(n_folds, shuffle=not contiguous)
    sequences, scores, deltas, splits = [], [], [], []
    for test, train in tqdm(splitter.split(trials_idx), desc="Fold", leave=False):
        nglm.traininds = trials_idx[train]
        sfs = mut.SequentialSelector(nglm, **seqsel_kwargs)
        direction = sfs.direction
        sfs.fit(**seqselfit_kwargs)
        if "full_scores" in seqselfit_kwargs and seqselfit_kwargs["full_scores"]:
            scores.append({"test": sfs.full_scores_test_, "train": sfs.full_scores_train_})
        else:
            scores.append({"test": sfs.scores_test_, "train": sfs.scores_train_})
        if direction == "backward":
            scores[-1]["basescores"] = {
                "test": sfs.basescores_test_,
                "train": sfs.basescores_train_,
            }
        if direction == "forward":
            deltas.append({"test": sfs.deltas_test_, "train": sfs.deltas_train_})
        else:
            deltas_test_ = pd.DataFrame(
                {
                    name: sfs.basescores_test_ - sfs.full_scores_test_[k].loc[:, 0]
                    for name, k in zip(design.covar.keys(), sfs.full_scores_test_.columns)
                }
            )
            deltas_train_ = pd.DataFrame(
                {
                    name: sfs.basescores_train_ - sfs.full_scores_train_[k].loc[:, 0]
                    for name, k in zip(design.covar.keys(), sfs.full_scores_train_.columns)
                }
            )

            deltas.append({"test": deltas_test_, "train": deltas_train_})
        sequences.append(sfs.sequences_)
        # TODO: Extract per-submodel alpha values
        splits.append({"test": trials_idx[test], "train": trials_idx[train]})
    outdict = {"scores": scores, "deltas": deltas, "sequences": sequences, "splits": splits}
    return outdict


def fit_stepwise_with_pseudoblocks(
    design,
    spk_t,
    spk_clu,
    binwidth,
    model,
    estimator,
    t_before,
    n_folds=5,
    contiguous=False,
    seqsel_kwargs={},
    seqselfit_kwargs={},
    n_impostors=100,
    **kwargs
):
    with parallel_backend(backend="loky", n_jobs=-1, inner_max_num_threads=2):
        print("context change worked")
        data_fit = fit_stepwise(
            design,
            spk_t,
            spk_clu,
            binwidth,
            model,
            estimator,
            n_folds,
            contiguous,
            seqsel_kwargs,
            seqselfit_kwargs,
        )
        null_fits = []
        while len(null_fits) < n_impostors:
            pseudoblock = pd.Series(
                generate_pseudo_blocks(design.base_df.shape[0]), index=design.base_df.index
            )
            pdesign = generate_design(design.base_df, pseudoblock, t_before, **kwargs)
            pfit = fit_stepwise(
                pdesign,
                spk_t,
                spk_clu,
                binwidth,
                model,
                estimator,
                n_folds,
                contiguous,
                seqsel_kwargs,
                seqselfit_kwargs,
            )
            null_fits.append(pfit)
    return data_fit, null_fits


def fit_impostor(
    design,
    impdf,
    spk_t,
    spk_clu,
    binwidth,
    model,
    estimator,
    t_before,
    n_impostors=100,
    n_folds=5,
    contiguous=False,
    **kwargs
):
    data_fit = fit(design, spk_t, spk_clu, binwidth, model, estimator, n_folds, contiguous)

    target_length = design.base_df.trial_end.max()
    null_fits = []
    while len(null_fits) < n_impostors:
        sampledf = sample_impostor(impdf, target_length, **kwargs)
        prior = sampledf["probabilityLeft"]
        try:
            pdesign = generate_design(sampledf, prior, t_before, **kwargs)
        except IndexError:
            continue
        pfit = fit(pdesign, spk_t, spk_clu, binwidth, model, estimator, n_folds, contiguous)
        null_fits.append(pfit)
    return data_fit, null_fits
