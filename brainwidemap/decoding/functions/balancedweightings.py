import numpy as np
import openturns
import pandas as pd
import pickle
from sklearn.utils.class_weight import compute_sample_weight
import torch

from brainbox.task.closed_loop import generate_pseudo_blocks, _draw_position, _draw_contrast

from behavior_models.models.utils import format_data as format_data_mut
from behavior_models.models.utils import format_input as format_input_mut
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside

from brainwidemap.decoding.functions.process_targets import optimal_Bayesian, check_bhv_fit_exists
from brainwidemap.decoding.functions.nulldistributions import generate_imposter_session
from brainwidemap.decoding.settings import modeldispatcher


def pdf_from_histogram(x, out):
    # unit test of pdf_from_histogram
    # out = np.histogram(np.array([0.9, 0.9]), bins=target_distribution[-1], density=True)
    # out[0][(np.array([0.9])[:, None] > out[1][None]).sum(axis=-1) - 1]
    return out[0][(x[:, None] > out[1][None]).sum(axis=-1) - 1]


def balanced_weighting(vec, continuous, use_openturns, bin_size_kde, target_distribution):
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.KernelSmoothing.html?highlight=kernel%20smoothing
    # This plug-in method for bandwidth estimation is based on the solve-the-equation rule from
    # (Sheather, Jones, 1991).
    if continuous:
        if use_openturns:
            factory = openturns.KernelSmoothing()
            sample = openturns.Sample(vec[:, None])
            bandwidth = factory.computePluginBandwidth(sample)
            distribution = factory.build(sample, bandwidth)
            proposal_weights = np.array(distribution.computePDF(sample)).squeeze()
            balanced_weight = np.ones(vec.size) / proposal_weights
        else:
            emp_distribution = np.histogram(vec, bins=target_distribution[-1], density=True)
            balanced_weight = pdf_from_histogram(vec, target_distribution) / pdf_from_histogram(
                vec, emp_distribution)
        #  plt.hist(y_train_inner[:, None], density=True)
        #  plt.plot(sample, proposal_weights, '+')
    else:
        balanced_weight = compute_sample_weight("balanced", y=vec)
    return balanced_weight


def get_target_pLeft(
        nb_trials, nb_sessions, take_out_unbiased, bin_size_kde, subjModel=None, antithetic=False):
    # if subjModel is empty, compute the optimal Bayesian prior
    if subjModel is not None:
        model = subjModel['modeltype'](subjModel['modelfit_path'].as_posix() + '/',
                                       subjModel['subjeids'],
                                       subjModel['subject'],
                                       actions=None,
                                       stimuli=None,
                                       stim_side=None)
        if subjModel['model_parameters'] is None:
            istrained, fullpath = check_bhv_fit_exists(subjModel['subject'], subjModel['modeltype'],
                                                       subjModel['subjeids'],
                                                       subjModel['modelfit_path'].as_posix() + '/',
                                                       modeldispatcher=subjModel['modeldispatcher'])
            if not istrained:
                raise ValueError('Something is wrong. The model should be trained by this line')
            model.load_or_train(loadpath=str(fullpath))
    else:
        model = None
    contrast_set = np.array([0., 0.0625, 0.125, 0.25, 1])
    target_pLeft = []
    for _ in np.arange(nb_sessions):
        if model is None or not subjModel['use_imposter_session_for_balancing']:
            pseudo_trials = pd.DataFrame()
            pseudo_trials['probabilityLeft'] = generate_pseudo_blocks(nb_trials)
            for i in range(pseudo_trials.shape[0]):
                position = _draw_position([-1, 1], pseudo_trials['probabilityLeft'][i])
                contrast = _draw_contrast(contrast_set, 'uniform')
                if position == -1:
                    pseudo_trials.loc[i, 'contrastLeft'] = contrast
                elif position == 1:
                    pseudo_trials.loc[i, 'contrastRight'] = contrast
                pseudo_trials.loc[i, 'stim_side'] = position
            pseudo_trials['signed_contrast'] = pseudo_trials['contrastRight']
            pseudo_trials.loc[pseudo_trials['signed_contrast'].isnull(),
                              'signed_contrast'] = -pseudo_trials['contrastLeft']
            pseudo_trials['choice'] = 1  # choice padding
        else:
            pseudo_trials = generate_imposter_session(subjModel['imposterdf'],
                                                      subjModel['eid'],
                                                      nb_trials,
                                                      nbSampledSess=10)
        side, stim, act, _ = format_data_mut(pseudo_trials)
        if model is None:
            msub_pseudo_tvec = optimal_Bayesian(act, side)
        elif not subjModel['use_imposter_session_for_balancing'] and model.name == modeldispatcher[expSmoothing_prevAction]:
            arr_params = (model.get_parameters(parameter_type='posterior_mean')[None] if subjModel['model_parameters'] is None
                          else np.array(list(subjModel['model_parameters'].values()))[None])
            valid = np.ones([1, pseudo_trials.index.size], dtype=bool)
            stim, _, side = format_input_mut([stim], [act], [side])
            act_sim, stim, side, msub_pseudo_tvec = model.simulate(arr_params,
                                                                   stim,
                                                                   side,
                                                                   torch.from_numpy(valid),
                                                                   nb_simul=10,
                                                                   only_perf=False,
                                                                   return_prior=True)
#            act_sim = act_sim.squeeze().T
#            stim = torch.tile(stim.squeeze()[None], (act_sim.shape[0], 1))
#            side = torch.tile(side.squeeze()[None], (act_sim.shape[0], 1))
#            msub_pseudo_tvec_ = model.compute_signal(
#                signal=('prior' if subjModel['target'] == 'pLeft' else subjModel['target']),
#                act=act_sim,
#                stim=stim,
#                side=side)
#            assert False
            msub_pseudo_tvec = msub_pseudo_tvec.squeeze().numpy()
        elif not subjModel['use_imposter_session_for_balancing'] and model.name == modeldispatcher[expSmoothing_stimside]:
            arr_params = (model.get_parameters(parameter_type='posterior_mean')[None] if subjModel['model_parameters'] is None
                          else np.array(list(subjModel['model_parameters'].values()))[None])
            valid = np.ones([1, pseudo_trials.index.size], dtype=bool)
            stim, act, side = format_input_mut([stim], [act], [side])
            output = model.evaluate(arr_params[None],
                                    return_details=True, act=act, stim=stim, side=side)
            msub_pseudo_tvec = output[1].squeeze()
        else:
            stim, act, side = format_input_mut([stim], [act], [side])
            msub_pseudo_tvec = model.compute_signal(
                signal=('prior' if subjModel['target'] == 'pLeft' else subjModel['target']),
                act=act,
                stim=stim,
                side=side)
            msub_pseudo_tvec = msub_pseudo_tvec['prior' if subjModel['target'] ==
                                                           'pLeft' else subjModel['target']]
            raise NotImplementedError('this is not supported since merge with wheel velocity')
        if take_out_unbiased:
            target_pLeft.append(
                msub_pseudo_tvec[(pseudo_trials.probabilityLeft != 0.5).values].ravel())
        else:
            target_pLeft.append(msub_pseudo_tvec.ravel())
    target_pLeft = np.concatenate(target_pLeft).ravel()
    if antithetic:
        target_pLeft = np.concatenate([target_pLeft, 1 - target_pLeft])
    out = np.histogram(target_pLeft,
                       bins=(np.arange(-bin_size_kde, 1 + bin_size_kde / 2., bin_size_kde) +
                             bin_size_kde / 2.),
                       density=True)
    return out, target_pLeft


def get_balanced_weighting(trials_df, metadata, **kwargs):
    if kwargs['balanced_weight'] and kwargs['balanced_continuous_target']:
        if not kwargs['use_imposter_session_for_balancing'] and (kwargs['model'] == optimal_Bayesian):
            target_distribution, _ = get_target_pLeft(nb_trials=trials_df.index.size,
                                                      nb_sessions=250,
                                                      take_out_unbiased=kwargs['no_unbias'],
                                                      bin_size_kde=kwargs['bin_size_kde'])
        else:
            subjModel = {
                'modeltype': kwargs['model'],
                'subjeids': metadata['eids_train'],
                'subject': metadata['subject'],
                'modelfit_path': kwargs['behfit_path'],
                'imposterdf': kwargs['imposterdf'],
                'use_imposter_session_for_balancing': kwargs['use_imposter_session_for_balancing'],
                'eid': metadata['eid'],
                'target': kwargs['target'],
                'model_parameters': kwargs['model_parameters'],
                'modeldispatcher': kwargs['modeldispatcher'],
            }
            target_distribution, allV_t = get_target_pLeft(
                nb_trials=trials_df.index.size,
                nb_sessions=250,
                take_out_unbiased=kwargs['no_unbias'],
                bin_size_kde=kwargs['bin_size_kde'],
                subjModel=subjModel)
    else:
        target_distribution = None

    return target_distribution
