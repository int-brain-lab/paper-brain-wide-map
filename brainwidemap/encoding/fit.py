# Third party libraries
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

# IBL libraries
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
        sfs.fit(**seqselfit_kwargs)
        if "full_scores" in seqselfit_kwargs and seqselfit_kwargs["full_scores"]:
            scores.append({"test": sfs.full_scores_test_, "train": sfs.full_scores_train_})
        else:
            scores.append({"test": sfs.scores_test_, "train": sfs.scores_train_})
        deltas.append({"test": sfs.deltas_test_, "train": sfs.deltas_train_})
        sequences.append(sfs.sequences_)
        # TODO: Extract per-submodel alpha values
        splits.append({"test": trials_idx[test], "train": trials_idx[train]})
    outdict = {"scores": scores, "deltas": deltas, "sequences": sequences, "splits": splits}
    return outdict


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
