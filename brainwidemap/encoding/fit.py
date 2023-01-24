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
from .design import generate_design


def fit(design, spk_t, spk_clu, binwidth, model, estimator, n_folds=5, contiguous=False, **kwargs):
    """
    Function to fit a model using a cross-validated design matrix.
    """
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
    """
    Fit a stepwise regression model using a cross-validated design matrix.

    This function will fit a regression model sequentially, adding or removing one group
    of covariates at a time to observe the impact of the change on the model score. The model
    can be fit either in a forward or backward direction, such that the full model is being built
    one regressor at a time or one regressor is being removed at a time. This relies on the 
    neurencoding.utils.SequentialSelector class, for which the seqsel_kwargs and seqselfit_kwargs
    are passed to the constructor and fit methods respectively.

    Parameters
    ----------
    design : neurencoding.DesignMatrix
        The design matrix for the model.
    spk_t : np.ndarray
        N x 1 array of spike times.
    spk_clu : np.ndarray
        N x 1 array of same shape as spk_t containing cluster IDs for each spike.
    binwidth : float
        Width of bins in which to place spikes
    model : neurencoding model class
        The type of model to fit, such as neurencoding.linear.LinearGLM
    estimator : sklearn estimator
        The estimator which will be passed to the model instance for actual fitting.
    n_folds : int, optional
        Number of cross validation folds, by default 5
    contiguous : bool, optional
        Whether or not to shuffle trial indices for cross validation, equivalent to
        the flip of sklearn's shuffle argument, by default False
    seqsel_kwargs : dict, optional
        Arguments to pass to neurencoding.SequentialSelector at construction, by default {}
    seqselfit_kwargs : dict, optional
        Arguments to pass to the `.fit` method of the SequentialSelector class, by default {}

    Returns
    -------
    dict
        Dictionary containing the following keys: ['scores', 'deltas', 'sequences', 'splits']:
            scores: list of dicts containing the test and train scores for each fold. If using
                backwards selection, the 'basescores' key will also be present, containing the
                scores for the model without any regressors removed.
            deltas: list of dicts containing the test and train deltas for each fold. This is
                the difference in score between the current model and the previous model fit. For
                the first regressor added in forward selection, this will be the same as the score
                of the model fit. For the first regressor removed in backwards selection, this will
                be the difference to the full model score.
            sequences: list of lists containing the sequence in which regressors were added or
                removed for each fold.
            splits: list of dicts containing the test and train indices for each fold.
    """
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
    """
    Wrapper around fit_stepwise that uses pseudoblocks on the prior term to estimate the effect
    of spurious correlations on the model score.

    Takes the same arguments (see docstring for fit_stepwise) except for the following:

    n_impostors : int, optional
        Number of pseudoblock fits to perform, by default 100
    
    returns a dictionary and a list of dictionaries, the first containing the results of the
    base model fit (see fit_stepwise), and the second containing the results of the pseudoblock
    control model fits.
    """
    with parallel_backend(backend="loky", n_jobs=-1, inner_max_num_threads=2):
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
