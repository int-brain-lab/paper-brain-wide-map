"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

# Standard library
import logging

# Third party libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# IBL libraries
import neurencoding.design_matrix as dm

_logger = logging.getLogger('brainwide')


def generate_design(trialsdf,
                    prior,
                    t_before,
                    bases,
                    iti_prior=[-0.4, -0.1],
                    fmove_offset=-0.4,
                    wheel_offset=-0.4,
                    contnorm=5.,
                    binwidth=0.02,
                    reduce_wheel_dim=True,
                    addtl_vars=None,
                    **kwargs):
    """
    Generate GLM design matrix object

    Parameters
    ----------
    trialsdf : pd.DataFrame
        Trials dataframe with trial timings in absolute (since session start) time
    prior : array-like
        Vector containing the prior estimate or true prior for each trial. Must be same length as
        trialsdf.
    t_before : float
        Time, in seconds, before stimulus onset that was used to define trial_start in trialsdf
    bases : dict
        Dictionary of basis functions for each regressor. Needs keys 'stim', 'feedback', 'fmove',
        (first movement) and 'wheel'.
    iti_prior : list, optional
        Two element list defining bounds on which step function for ITI prior is
        applied, by default [-0.4, -0.1]
    contnorm : float, optional
        Normalization factor for contrast, by default 5.
    binwidth : float, optional
        Size of bins to use for design matrix, in seconds, by default 0.02
    """
    if len(kwargs) > 0:
        _logger.info(f"keys {kwargs.keys()} were not used in generate_design,"
                     " despite being passed.")
    trialsdf['adj_contrastL'] = np.tanh(contnorm * trialsdf['contrastLeft']) / np.tanh(contnorm)
    trialsdf['adj_contrastR'] = np.tanh(contnorm * trialsdf['contrastRight']) / np.tanh(contnorm)
    trialsdf['prior'] = prior
    trialsdf['prior_last'] = pd.Series(np.roll(trialsdf['prior'], 1), index=trialsdf.index)
    trialsdf['pLeft_last'] = pd.Series(np.roll(trialsdf['probabilityLeft'], 1),
                                       index=trialsdf.index)

    vartypes = {
        'choice': 'value',
        'response_times': 'timing',
        'probabilityLeft': 'value',
        'pLeft_last': 'value',
        'feedbackType': 'value',
        'feedback_times': 'timing',
        'contrastLeft': 'value',
        'adj_contrastL': 'value',
        'contrastRight': 'value',
        'adj_contrastR': 'value',
        'goCue_times': 'timing',
        'stimOn_times': 'timing',
        'trial_start': 'timing',
        'trial_end': 'timing',
        'prior': 'value',
        'prior_last': 'value',
        'wheel_velocity': 'continuous',
        'firstMovement_times': 'timing'
    }
    if addtl_vars is not None and isinstance(addtl_vars, dict):
        vartypes.update(addtl_vars)

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        stepvec[stepbounds[0]:stepbounds[1]] = row.prior_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.prior_last
        zerovec[currtr_end:] = row.prior
        return zerovec

    design = dm.DesignMatrix(trialsdf, vartypes, binwidth=binwidth)
    stepbounds = [design.binf(t_before + iti_prior[0]), design.binf(t_before + iti_prior[1])]

    design.add_covariate_timing('stimonL',
                                'stimOn_times',
                                bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastL',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR',
                                'stimOn_times',
                                bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastR',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_timing('correct',
                                'feedback_times',
                                bases['feedback'],
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect',
                                'feedback_times',
                                bases['feedback'],
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    design.add_covariate_timing('fmoveL',
                                'firstMovement_times',
                                bases['fmove'],
                                offset=fmove_offset,
                                cond=lambda tr: tr.choice == 1,
                                desc='Lead up to first movement leading to left choice')
    design.add_covariate_timing('fmoveR',
                                'firstMovement_times',
                                bases['fmove'],
                                offset=fmove_offset,
                                cond=lambda tr: tr.choice == -1,
                                desc='Lead up to first movement leading to right choice')

    design.add_covariate_raw('pLeft', stepfunc_prestim, desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr',
                             stepfunc_poststim,
                             desc='Step function on post-stimulus prior estimate')

    design.add_covariate('wheel', trialsdf['wheel_velocity'], bases['wheel'], wheel_offset)
    design.compile_design_matrix()

    if reduce_wheel_dim:
        _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']], full_matrices=False)
        variances = s**2 / (s**2).sum()
        n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
        wheelcols = design[:, design.covar['wheel']['dmcol_idx']]
        reduced = wheelcols @ v[:n_keep].T
        bases_reduced = bases['wheel'] @ v[:n_keep].T
        keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
        basedm = design[:, keepcols]
        design.dm = np.hstack([basedm, reduced])
        design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
        design.covar['wheel']['bases'] = bases_reduced

    print('Condition of design matrix:', np.linalg.cond(design.dm))
    return design


def sample_impostor(impdf,
                    target_length,
                    timing_vars=[
                        'stimOn_times', 'goCue_times', 'firstMovement_times', 'feedback_times',
                        'trial_start', 'trial_end'
                    ],
                    iti_generator=norm(loc=0.5, scale=0.2),
                    verif_binwidth=None,
                    maxeps_corr=1e-9,
                    **kwargs):
    """
    Samples an impostor session below given length from a import dataframe file provided

    Parameters
    ----------
    impdf : pandas.DataFrame
        The impostor dataframe to sample from, consisting of many trials in relative time. This
        means that the trial_start column must be all zeros, and each other timing column is time
        relative to trial start.
    target_length : float
        The session length to sample from the impostor dataframe, usually the length of your data
    timing_vars : list of str
        Column names in impdf which are timing variables, and therefore must be adjusted to reflect
        position relative to one another in time.
    iti_generator : scip.stats.rv_continuous subclass instance
        Distribution used to generate the inter-trial intervals for the sampled dataframe. Must
        have an .rvs method taking a size= argument.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing impostor trials along with the information per trial
    """
    # Impostor dataframe subsampling
    rng = np.random.default_rng()
    # Choose a random starting point in the main impostor df and roll to start there
    startind = rng.choice(impdf.index, size=1)
    rolldf = impdf.reindex(np.roll(impdf.index, -startind)).reset_index(drop=True)
    # Compute cumulative sum of trial durations and find last trial under our target length
    dursum = rolldf['duration'].cumsum().values
    endind = np.argwhere(dursum < target_length).flat[-1]
    sampledf = rolldf.loc[:endind].copy()  # subsample our rolled df to the number of trials

    # Generate synthetic ITIs between start and end of trials. This will put our subsampled df
    # over the target length again, but it's still more efficient than computing N_master_trials
    # random ITIs and cumsum.
    iti = iti_generator.rvs(size=endind + 1)
    iti[iti < 0] = 0
    endings = sampledf['trial_end'].reindex(np.roll(sampledf.index, 1)).reset_index(drop=True)
    endings.at[0] = 0
    endings = endings + iti
    endlast = endings.cumsum()

    sampledf.loc[:, timing_vars] = sampledf.loc[:, timing_vars].add(endlast, axis=0)
    sampledf.drop(columns=['orig_eid', 'duration'], inplace=True)

    # Because floating point math is hell and I have been condemned to damnation
    if verif_binwidth is not None:
        if not isinstance(verif_binwidth, float):
            _logger.warning("Verification binwidth for sample_impostor wasn't a float. Ignoring.")
        else:

            def binf(t):
                return np.ceil(t / verif_binwidth).astype(int)

            sampledf['newdur'] = sampledf['trial_end'] - sampledf['trial_start']
            diffs = sampledf.apply(lambda row: binf(row.newdur) - row.wheel_velocity.shape[0],
                                   axis=1)
            badinds = diffs.index[diffs != 0]
            if np.any(diffs.abs() > 1):
                raise IndexError('Error in wheel velocity vs trial duration during sampling.')
            for idx in badinds:
                direction = diffs.loc[idx]
                offset = -1 if direction < 0 else 0
                target = (sampledf.loc[idx].wheel_velocity.shape[0] + offset) * verif_binwidth
                eps = sampledf.newdur.loc[idx] - target
                _logger.info(f'Trial {idx} trail_end offset by {eps} due to mismatch of same size')
                if eps < maxeps_corr:
                    sampledf.loc[idx, 'trial_end'] += -(eps + direction * maxeps_corr)
        sampledf.drop(columns=['newdur'], inplace=True)
    # Finally, remove all those trials where we have exceeded the target limit thanks to the
    # ITI sampling
    maxind = np.searchsorted(sampledf.trial_end, target_length)
    return sampledf.iloc[:maxind + 1]
