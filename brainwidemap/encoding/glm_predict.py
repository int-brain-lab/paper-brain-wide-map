"""
Functions for examining the predictions of a fit GLM model against the actual data.
"""
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

# IBL libraries
from brainbox.plot import peri_event_time_histogram


def predict(
    nglm, targ_regressors=None, trials=None, retlab=False, incl_bias=True, glm_type="linear"
):
    """
    Predict a given set of regressors for a given set of trials.

    Parameters
    ----------
    nglm : neurencoding model
        Fit GLM model.
    targ_regressors : list, optional
        List of regressors in the model which will be included in the prediction. None causes all
        regressors to be included, by default None
    trials : array-like, optional
        List of trial numbers to include in prediction, passing None causes all trials to be
        used in computing the prediction, by default None
    retlab : bool, optional
        Whether to return the trial labels for each time bin in the prediction, by default False
    incl_bias : bool, optional
        Whether or not to include the intercept or bias term fit to the data in
        predictions, by default True
    glm_type : str, optional
        Type of glm fit, determining the link function for predictions. Can be either 'linear' or
        'poisson', by default "linear"

    Returns
    -------
    dict, np.ndarray (optional)
        Returns dict of cluster IDs and predicted spike counts for each trial. If retlab is True,
        also returns the trial labels for each time bin in the prediction.

    """
    if trials is None:
        trials = nglm.design.trialsdf.index
    if targ_regressors is None:
        targ_regressors = nglm.design.covar.keys()
    dmcols = np.hstack([nglm.design.covar[r]["dmcol_idx"] for r in targ_regressors])
    dmcols = np.sort(dmcols)
    trlabels = nglm.design.trlabels
    trfilter = np.isin(trlabels, trials).flatten()
    w = nglm.coefs
    b = nglm.intercepts
    dm = nglm.design.dm[trfilter, :][:, dmcols]
    if glm_type == "poisson":
        link = np.exp
    elif glm_type == "linear":

        def link(x):
            return x

    else:
        raise TypeError("nglm must be poisson or linear")
    if incl_bias:
        pred = {cell: link(dm @ w.loc[cell][dmcols] + b.loc[cell]) for cell in w.index}
    else:
        pred = {cell: link(dm @ w.loc[cell][dmcols]) for cell in w.index}
    if not retlab:
        return pred
    else:
        return pred, trlabels[trfilter].flatten()


def pred_psth(
    nglm, align_time, t_before, t_after, targ_regressors=None, trials=None, incl_bias=True
):
    """
    Compute a peri-event-time-histogram of predicted firing for a given alignment event.

    Parameters
    ----------
    nglm : neurencoding model
        Fit GLM model.
    align_time : str
        Column in the design trials dataframe to align to.
    t_before : float
        Time in seconds before the event to include in PETH
    t_after : float
        Time in seconds after the event to include in PETH
    targ_regressors : list, optional
        List of regressors in the model to include in generated prediction, by default None
    trials : array-like, optional
        List of trials on which to compute the prediction. None causes all trials to be
        included, by default None
    incl_bias : bool, optional
        Whether or not to include intercept in predictions, by default True

    Returns
    -------
    dict
        Dictionary of cell, peth pairs.
    """
    if trials is None:
        trials = nglm.design.trialsdf.index
    designtimes = nglm.design.trialsdf[align_time]
    times = designtimes[np.isfinite(designtimes)].apply(nglm.binf)
    tbef_bin = nglm.binf(t_before)
    taft_bin = nglm.binf(t_after)
    pred, labels = predict(nglm, targ_regressors, trials, retlab=True, incl_bias=incl_bias)
    t_inds = [np.searchsorted(labels, tr) + times[tr] for tr in trials]
    winds = [(t - tbef_bin, t + taft_bin) for t in t_inds]
    psths = {}
    for cell in pred.keys():
        cellpred = pred[cell]
        windarr = np.vstack([cellpred[w[0] : w[1]] for w in winds])
        psths[cell] = (
            np.mean(windarr, axis=0) / nglm.binwidth,
            np.std(windarr, axis=0) / nglm.binwidth,
        )
    return psths


class GLMPredictor:
    """
    Class to generate PETHs for a given GLM model and spiking data, and plot.
    """

    def __init__(self, trialsdf, nglm, spk_t, spk_clu):
        """
        Generate a predictor object, which can then use the internally stored model and spikes to
        compare predictions from the model against the actual observed spiking data.

        Parameters
        ----------
        trialsdf : pd.DataFrame
            Trials dataframe used in the fit GLM model, with columns for trial start and end times.
        nglm : neurencoding model
            Fit GLM model.
        spk_t : np.ndarray
            N x 1 array of spike times for all clusters
        spk_clu : np.ndarray
            N x 1 array of labels for each spike time identifying the cluster to which it belongs.
        """
        self.covar = list(nglm.design.covar.keys())
        self.nglm = nglm
        self.binnedspikes = nglm.binnedspikes
        self.design = nglm.design
        self.spk_t = spk_t
        self.spk_clu = spk_clu
        self.trials = trialsdf.index
        self.trialsdf = trialsdf  # maybe not best way to do this
        self.full_psths = {}
        self.cov_psths = {}
        self.combweights = nglm.combine_weights()

    def psth_summary(self, align_time, unit, t_before=0.1, t_after=0.6, trials=None, ax=None):
        """
        Generate a summary plot of the actual and predicted firing rates for a given unit. Will
        produce 3 separate subplots:
            1. Actual and predicted PETHs for the given event
            2. The contributions of each set of regressors to the prediction, which sum to the
                prediction in (1)
            3. The actual weights for each regressor, combined across basis functions where
                appropriate. Will not plot the some terms like boxcar terms or offsets.

        Optionally allows for a subset of trials to be used in the plot, useful for e.g. plotting
        predictions on a subset of trials such as zero contrast trials.

        Parameters
        ----------
        align_time : str
            Column in the original trialsdf passed to the instance to align to.
        unit : int
            Integer label of the unit to plot.
        t_before : float, optional
            Time before the align time to plot, by default 0.1
        t_after : float, optional
            Time after the align time to plot, by default 0.6
        trials : array-like, optional
            List of trials on which the prediction and PETH will be computed., by default None
        ax : matplotlib.pyplot axes array, optional
            Array of 3 axes on which to place plots. Must be a one-dimensional array of length 3.

        Returns
        -------
        matplotlib.pyplot axes
            Axes on which the prediction and prediction breakdown are plotted.
        """
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(8, 12))

        if trials is None:
            trials = self.trials
        times = self.trialsdf.loc[trials, align_time]  #
        peri_event_time_histogram(
            self.spk_t,
            self.spk_clu,
            times[np.isfinite(times)],
            unit,
            t_before,
            t_after,
            bin_size=self.nglm.binwidth,
            error_bars="sem",
            ax=ax[0],
            smoothing=0.01,
        )
        keytuple = (align_time, t_before, t_after, tuple(trials))
        if keytuple not in self.full_psths:
            self.full_psths[keytuple] = pred_psth(
                self.nglm, align_time, t_before, t_after, trials=trials
            )
            self.cov_psths[keytuple] = {}
            tmp = self.cov_psths[keytuple]
            for cov in self.covar:
                tmp[cov] = pred_psth(
                    self.nglm,
                    align_time,
                    t_before,
                    t_after,
                    targ_regressors=[cov],
                    trials=trials,
                    incl_bias=False,
                )
        for cov in self.covar:
            ax[2].plot(self.combweights[cov].loc[unit])
        ax[2].set_title("Individual kernels (not PSTH contrib)")
        x = np.arange(-t_before, t_after, self.nglm.binwidth)
        ax[0].step(
            x,
            self.full_psths[keytuple][unit][0],
            where="post",
            color="orange",
            label="Model prediction",
        )
        ax[0].legend()
        for cov in self.covar:
            ax[1].step(x, self.cov_psths[keytuple][cov][unit][0], where="post", label=cov)
        ax[1].set_title("Individual component contributions")
        ax[1].legend()
        if hasattr(self.nglm, "clu_regions"):
            unitregion = self.nglm.clu_regions[unit]
            plt.suptitle(f"Unit {unit} from region {unitregion}")
        else:
            plt.suptitle(f"Unit {unit}")
        plt.tight_layout()
        return ax
