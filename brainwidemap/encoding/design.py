"""
This file contains the main model design matrix generator for all of the linear model fits.
Changes to the model specification, beyond basis function choices, should be made here.
"""

# Standard library
import logging

# IBL libraries
import neurencoding.design_matrix as dm

# Third party libraries
import numpy as np
import pandas as pd

_logger = logging.getLogger("brainwide")


def generate_design(
    trialsdf,
    prior,
    t_before,
    bases,
    iti_prior_win=[-0.4, -0.1],
    iti_prior_val=None,
    fmove_offset=-0.4,
    wheel_offset=-0.4,
    contnorm=5.0,
    binwidth=0.02,
    reduce_wheel_dim=True,
    addtl_vars=None,
    **kwargs,
):
    """
    Generate GLM design matrix object

    Parameters
    ----------
    trialsdf : pd.DataFrame
        Trials dataframe with trial timings in absolute (since session start) time
    prior : array-like
        Vector containing the prior estimate or true prior for each trial. Must be same length as
        trialsdf. If iti_prior_val is not None, this is used as the prior for the ITI period.
    t_before : float
        Time, in seconds, before stimulus onset that was used to define trial_start in trialsdf.
        This defines the beginning of the window, relative to each trial, where we examine the
        spiking data.
    bases : dict
        Dictionary of basis functions for each regressor. Needs keys 'stim', 'feedback', 'fmove',
        (first movement) and 'wheel'.
    iti_prior_win : list, optional
        Two element list defining bounds relative to stimulus on which step function for ITI prior
        is applied, by default [-0.4, -0.1]
    iti_prior_val : float, optional
        Value to use for ITI prior, by default None. If None, uses prior.
    fmove_offset : float, optional
        Offset, in seconds, to apply to first movement regressor, by default -0.4. This is relative
        to the movement onset time, and if you want a purely anti-causal kernel should be
        equivalent to the basis function length.
    wheel_offset : float, optional
        Offset, in seconds, to apply to wheel regressor, by default -0.4. See above.
    contnorm : float, optional
        Normalization factor for contrast, by default 5.
        Applied as tanh(contrast * contnorm) / tanh(contnorm)
    binwidth : float, optional
        Size of bins to use for design matrix, in seconds, by default 0.02. This must match
        the binwidth of the neural glm object later used to fit the design matrix.
    reduce_wheel_dim : bool, optional
        Whether to reduce the dimensionality of the wheel regressor, by default True. If True,
        will use enough principal components to capture 99.999% of the variance. Smooths out
        very high frequency fluctuations in the wheel velocity.
    addtl_vars : dict, optional
        Dictionary of additional variables in the trialsdf along with their variable types. These
        are columns that are not normally part of the output of load_trials_df, and will therefore
        raise an error if the design matrix building doesn't find a type specification for them.
        See neurencoding.design.DesignMatrix for more details on variable types, by default None.
    """
    if len(kwargs) > 0:
        _logger.info(
            f"keys {kwargs.keys()} were not used in generate_design," " despite being passed."
        )
    trialsdf["adj_contrastL"] = np.tanh(contnorm * trialsdf["contrastLeft"]) / np.tanh(contnorm)
    trialsdf["adj_contrastR"] = np.tanh(contnorm * trialsdf["contrastRight"]) / np.tanh(contnorm)
    trialsdf["prior"] = prior
    if iti_prior_val is not None:
        trialsdf["iti_prior"] = iti_prior_val
    else:
        trialsdf["iti_prior"] = prior
    trialsdf["prior_last"] = pd.Series(np.roll(trialsdf["prior"], 1), index=trialsdf.index)
    trialsdf["iti_prior_last"] = pd.Series(np.roll(trialsdf["iti_prior"], 1), index=trialsdf.index)
    trialsdf["pLeft_last"] = pd.Series(
        np.roll(trialsdf["probabilityLeft"], 1), index=trialsdf.index
    )

    vartypes = {
        "choice": "value",
        "response_times": "timing",
        "probabilityLeft": "value",
        "pLeft_last": "value",
        "feedbackType": "value",
        "feedback_times": "timing",
        "contrastLeft": "value",
        "adj_contrastL": "value",
        "contrastRight": "value",
        "adj_contrastR": "value",
        "goCue_times": "timing",
        "stimOn_times": "timing",
        "trial_start": "timing",
        "trial_end": "timing",
        "prior": "value",
        "prior_last": "value",
        "iti_prior": "value",
        "iti_prior_last": "value",
        "wheel_velocity": "continuous",
        "firstMovement_times": "timing",
    }
    if addtl_vars is not None and isinstance(addtl_vars, dict):
        vartypes.update(addtl_vars)

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0] : stepbounds[1]] = row.iti_prior_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.prior_last
        zerovec[currtr_end:] = row.prior
        return zerovec

    design = dm.DesignMatrix(trialsdf, vartypes, binwidth=binwidth)
    stepbounds = [
        design.binf(t_before + iti_prior_win[0]),
        design.binf(t_before + iti_prior_win[1]),
    ]

    design.add_covariate_timing(
        "stimonL",
        "stimOn_times",
        bases["stim"],
        cond=lambda tr: np.isfinite(tr.contrastLeft),
        deltaval="adj_contrastL",
        desc="Kernel conditioned on L stimulus onset",
    )
    design.add_covariate_timing(
        "stimonR",
        "stimOn_times",
        bases["stim"],
        cond=lambda tr: np.isfinite(tr.contrastRight),
        deltaval="adj_contrastR",
        desc="Kernel conditioned on R stimulus onset",
    )
    design.add_covariate_timing(
        "correct",
        "feedback_times",
        bases["feedback"],
        cond=lambda tr: tr.feedbackType == 1,
        desc="Kernel conditioned on correct feedback",
    )
    design.add_covariate_timing(
        "incorrect",
        "feedback_times",
        bases["feedback"],
        cond=lambda tr: tr.feedbackType == -1,
        desc="Kernel conditioned on incorrect feedback",
    )
    design.add_covariate_timing(
        "fmoveL",
        "firstMovement_times",
        bases["fmove"],
        offset=fmove_offset,
        cond=lambda tr: tr.choice == 1,
        desc="Lead up to first movement leading to left choice",
    )
    design.add_covariate_timing(
        "fmoveR",
        "firstMovement_times",
        bases["fmove"],
        offset=fmove_offset,
        cond=lambda tr: tr.choice == -1,
        desc="Lead up to first movement leading to right choice",
    )

    design.add_covariate_raw("pLeft", stepfunc_prestim, desc="Step function on prior estimate")
    design.add_covariate_raw(
        "pLeft_tr", stepfunc_poststim, desc="Step function on post-stimulus prior estimate"
    )

    design.add_covariate("wheel", trialsdf["wheel_velocity"], bases["wheel"], wheel_offset)
    design.compile_design_matrix()

    if reduce_wheel_dim:
        _, s, v = np.linalg.svd(design[:, design.covar["wheel"]["dmcol_idx"]], full_matrices=False)
        variances = s**2 / (s**2).sum()
        n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
        wheelcols = design[:, design.covar["wheel"]["dmcol_idx"]]
        reduced = wheelcols @ v[:n_keep].T
        bases_reduced = bases["wheel"] @ v[:n_keep].T
        keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar["wheel"]["dmcol_idx"])
        basedm = design[:, keepcols]
        design.dm = np.hstack([basedm, reduced])
        design.covar["wheel"]["dmcol_idx"] = design.covar["wheel"]["dmcol_idx"][:n_keep]
        design.covar["wheel"]["bases"] = bases_reduced

    _logger.info(f"Condition of design matrix: {np.linalg.cond(design.dm)}")
    return design
