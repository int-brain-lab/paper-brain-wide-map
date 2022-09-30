import numpy as np
import pandas as pd
import brainbox.behavior.pyschofit as pfit

'''
trialsdf_neurometric = nb_trialsdf.reset_index() if (pseudo_id == -1) else \
    pseudosess[pseudomask].reset_index()
if kwargs['model'] is not None:
    blockprob_neurometric = dut.compute_target(
        'pLeft',
        subject,
        subjeids,
        eid,
        kwargs['modelfit_path'],
        binarization_value=kwargs['binarization_value'],
        modeltype=kwargs['model'],
        beh_data_test=trialsdf if pseudo_id == -1 else pseudosess,
        behavior_data_train=behavior_data_train,)

    trialsdf_neurometric['blockprob_neurometric'] = np.stack([
        np.greater_equal(
            blockprob_neurometric[(mask & (trialsdf.choice != 0) if pseudo_id
                                                                    == -1 else pseudomask)], border).astype(int)
        for border in kwargs['border_quantiles_neurometric']
    ]).sum(0)

else:
    blockprob_neurometric = trialsdf_neurometric['probabilityLeft'].replace(
        0.2, 0).replace(0.8, 1)
    trialsdf_neurometric['blockprob_neurometric'] = blockprob_neurometric
'''

def get_target_df(target, pred, test_idxs, trialsdf):
    """
    Get two arrays needed for the psychofit method for fitting an erf.
    Parameters
    ----------
    eid : str
        uuid of session
    target : numpy.ndarray
        vector of all targets for the given session
    pred : numpy.ndarray
        vector of all predictions for the given session and region
    test_idxs : numpy.ndarray
        indices of the used in the test set and scoring computation
    Returns
    -------
    tuple of numpy.ndarrays
        two 3 x 9 arrays, the first of which is for P(Left) = 0.2 ,the latter for 0.8, with
        the specifications required for mle_fit_psycho.
    """
    offset = 1 - target.max()
    test_target = target[test_idxs]
    test_pred = pred[test_idxs]
    corr_test = test_target + offset
    corr_pred = test_pred + offset
    pred_signs = np.sign(corr_pred)
    df = pd.DataFrame({'stimuli': corr_test,
                       'predictions': corr_pred,
                       'sign': pred_signs,
                       'blockprob': trialsdf.loc[test_idxs, 'blockprob_neurometric']},
                      index=test_idxs)
    grpby = df.groupby(['blockprob', 'stimuli'])
    grpbyagg = grpby.agg({'sign': [('num_trials', 'count'),
                                   ('prop_L', lambda x: ((x == 1).sum() + (x == 0).sum() / 2.) / len(x))]})
    return [grpbyagg.loc[k].reset_index().values.T for k in
            grpbyagg.index.get_level_values('blockprob').unique().sort_values()]


def get_neurometric_parameters_(prob_arr, possible_contrasts, force_positive_neuro_slopes=False, nfits=100):
    if force_positive_neuro_slopes:
        pars, L = pfit.mle_fit_psycho(prob_arr,
                                      P_model='erf_psycho_2gammas',
                                      nfits=nfits)
    else:
        pars, L = pfit.mle_fit_psycho(prob_arr,
                                      P_model='erf_psycho_2gammas',
                                      nfits=nfits,
                                      parmin=np.array([-1., -10., 0., 0.]),
                                      parmax=np.array([1., 10., 0.4, 0.4]))
    contrasts = prob_arr[0, :] if possible_contrasts is None else possible_contrasts
    fit_trace = pfit.erf_psycho_2gammas(pars, contrasts)
    range = fit_trace[np.argwhere(np.isclose(contrasts, 1)).flat[0]] - \
            fit_trace[np.argwhere(np.isclose(contrasts, -1)).flat[0]]
    slope = pars[1]
    zerind = np.argwhere(np.isclose(contrasts, 0)).flat[0]
    return {'range': range, 'slope': slope, 'pars': pars, 'L': L, 'fit_trace': fit_trace, 'zerind': zerind}

def fit_get_shift_range(prob_arrs,
                        force_positive_neuro_slopes=False,
                        seed_=None,
                        possible_contrasts=np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1]),
                        nfits=100,
                        ):
    """
    Fit psychometric functions with erf with two gammas for lapse rate, and return the parameters,
    traces, and the slope and shift.
    Parameters
    ----------
    lowprob_arr
    seed_
    possible_contrasts
    lowprob_arr : numpy.ndarray
        3 x 9 array for function fitting (see get_target_df), from low probability left block
    highprob_arr : numpy.ndarray
        Same as above, for high probability Left
    """
    # pLeft = 0.2 blocks
    if seed_ is not None:
        np.random.seed(seed_)
    lows = get_neurometric_parameters_(prob_arrs[0], possible_contrasts,
                                       force_positive_neuro_slopes=force_positive_neuro_slopes, nfits=nfits)
    # pLeft = 0.8 blocks
    if seed_ is not None:
        np.random.seed(seed_)
    highs = get_neurometric_parameters_(prob_arrs[-1], possible_contrasts,
                                        force_positive_neuro_slopes=force_positive_neuro_slopes, nfits=nfits)

    # compute shift
    shift = highs['fit_trace'][highs['zerind']] - lows['fit_trace'][lows['zerind']]
    params = {'low_pars': lows['pars'], 'low_likelihood': lows['L'],
              'high_pars': highs['pars'], 'high_likelihood': highs['L'],
              'low_fit_trace': lows['fit_trace'], 'high_fit_trace': highs['fit_trace'],
              'low_slope': lows['slope'], 'high_slope': highs['slope'], 'low_range': lows['range'],
              'high_range': highs['range'], 'shift': shift,
              'mean_range': (lows['range'] + highs['range'] ) /2.,
              'mean_slope': (lows['slope'] + highs['slope'] ) /2.}

    params = {**params, **{'NB_QUANTILES': len(prob_arrs)}}
    for (out, k) in zip([lows, highs], [0, len(prob_arrs) - 1]):
        params = {**params, **{'quantile_%i_0contrastLevel' % k: out['fit_trace'][out['zerind']],
                               'quantile_%i_pars' % k: out['pars'],
                               'quantile_%i_likelihood' % k: out['L'],
                               'quantile_%i_fit_trace' % k: out['fit_trace'],
                               'quantile_%i_slope' % k: out['slope'],
                               'quantile_%i_range' % k: out['range']}}

    if len(prob_arrs) > 2:
        for k in range(1, len(prob_arrs) - 1):
            mediums = get_neurometric_parameters_(prob_arrs[k], possible_contrasts,
                                                  force_positive_neuro_slopes=force_positive_neuro_slopes)
            params = {**params, **{'quantile_%i_0contrastLevel' % k: mediums['fit_trace'][mediums['zerind']],
                                   'quantile_%i_pars' % k: mediums['pars'],
                                   'quantile_%i_likelihood' % k: mediums['L'],
                                   'quantile_%i_fit_trace' % k: mediums['fit_trace'],
                                   'quantile_%i_slope' % k: mediums['slope'],
                                   'quantile_%i_range' % k: mediums['range']}}
    return params

def get_neurometric_parameters(fit_result, trialsdf, compute_on_each_fold, force_positive_neuro_slopes):
    # fold-wise neurometric curve
    if compute_on_each_fold:
        raise NotImplementedError('Sorry, this is not up to date to perform computation on each folds. Ask Charles F.')
    if compute_on_each_fold:
        try:
            prob_arrays = [get_target_df(fit_result['target'],
                                         fit_result['predictions'][k],
                                         fit_result['idxes_test'][k],
                                         trialsdf)
                           for k in range(fit_result['nFolds'])]

            fold_neurometric = [fit_get_shift_range(prob_arrays[k][0], prob_arrays[k][1], force_positive_neuro_slopes)
                                for k in range(fit_result['nFolds'])]
        except KeyError:
            fold_neurometric = None
    else:
        fold_neurometric = None

    # full neurometric curve
    full_test_prediction = np.zeros(len(fit_result['target']))
    for k in range(fit_result['nFolds']):
        full_test_prediction[fit_result['idxes_test'][k]] = fit_result['predictions_test'][k]

    prob_arrs = get_target_df(fit_result['target'], full_test_prediction,
                              np.arange(len(fit_result['target'])), trialsdf)
    full_neurometric = fit_get_shift_range(prob_arrs, force_positive_neuro_slopes)

    return full_neurometric, fold_neurometric
