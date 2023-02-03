import numpy as np
import torch
from behavior_models.models.utils import format_input as mut_format_input
from brainbox.task.closed_loop import generate_pseudo_session
from brainwidemap.decoding.functions.process_targets import optimal_Bayesian, check_bhv_fit_exists


def generate_null_distribution_session(trials_df, metadata, **kwargs):
    if 'signedContrast' in trials_df.columns:
        out = np.nan_to_num(trials_df.contrastLeft.values) - np.nan_to_num(trials_df.contrastRight.values)
        assert(np.all(np.nan_to_num(trials_df.signedContrast.values) == out))
    if kwargs['use_imposter_session']:
        if not kwargs['constrain_null_session_with_beh']:
            pseudosess = generate_imposter_session(kwargs['imposterdf'],
                                                   metadata['eid'],
                                                   trials_df.index.size,
                                                   nbSampledSess=5)
        else:
            feedback_0contrast = trials_df[(trials_df.contrastLeft == 0).values + (
                    trials_df.contrastRight == 0).values].feedbackType.mean()

            pseudosess_s = [
                generate_imposter_session(kwargs['imposterdf'],
                                          metadata['eid'],
                                          trials_df.index.size,
                                          nbSampledSess=10) for _ in range(50)
            ]
            feedback_pseudo_0cont = [
                pseudo[(pseudo.contrastLeft == 0).values +
                       (pseudo.contrastRight == 0).values].feedbackType.mean()
                for pseudo in pseudosess_s
            ]
            pseudosess = pseudosess_s[np.argmin(
                np.abs(np.array(feedback_pseudo_0cont) - feedback_0contrast))]
    else:
        pseudosess = generate_pseudo_session(trials_df, generate_choices=False)
        if kwargs['model'] is not None and kwargs['model'] != optimal_Bayesian and kwargs['model'].name == 'actKernel':
            subjModel = {
                **metadata,
                'modeltype': kwargs['model'],
                'behfit_path': kwargs['behfit_path'],
            }
            pseudosess['choice'] = generate_choices(
                pseudosess, trials_df, subjModel,
                kwargs['modeldispatcher'],
                kwargs['constrain_null_session_with_beh'],
                kwargs['model_parameters'])
            pseudosess['feedbackType'] = np.where(pseudosess['choice'] == pseudosess['stim_side'], 1, -1)
        else:
            pseudosess['choice'] = trials_df.choice
    return pseudosess


def generate_choices(pseudosess, trials_df, subjModel, modeldispatcher, constrain_null_session_with_beh, model_parameters=None):

    if model_parameters is None:
        istrained, fullpath = check_bhv_fit_exists(subjModel['subject'], subjModel['modeltype'],
                                                   subjModel['eids_train'],
                                                   subjModel['behfit_path'].as_posix() + '/',
                                                   modeldispatcher, single_zeta=True)
    else:
        istrained, fullpath = True, ''

    if not istrained:
        raise ValueError('Something is wrong. The model should be trained by this line')
    model = subjModel['modeltype'](subjModel['behfit_path'],
                                   subjModel['eids_train'],
                                   subjModel['subject'],
                                   actions=None,
                                   stimuli=None,
                                   stim_side=None,
                                   single_zeta=True)

    if model_parameters is None:
        model.load_or_train(loadpath=str(fullpath))
        arr_params = model.get_parameters(parameter_type='posterior_mean')[None]
    else:
        arr_params = np.array(list(model_parameters.values()))[None]
    valid = np.ones([1, pseudosess.index.size], dtype=bool)
    stim, _, side = mut_format_input([pseudosess.signed_contrast.values],
                                     [trials_df.choice.values], [pseudosess.stim_side.values])
    act_sim, stim, side = model.simulate(arr_params,
                                         stim,
                                         side,
                                         torch.from_numpy(valid),
                                         nb_simul=10000 if constrain_null_session_with_beh else 1,
                                         only_perf=False)
    act_sim = np.array(act_sim.squeeze().T, dtype=np.int64)

    if constrain_null_session_with_beh:
        # behavior on simulations
        perf_1contrast_sims = np.array([torch.mean((torch.from_numpy(a_s) == side.squeeze())[stim.abs().squeeze() == 1] * 1.).numpy()
                                        for a_s in act_sim])
        perf_0contrast_sims = np.array([torch.mean((torch.from_numpy(a_s) == side.squeeze())[stim.squeeze() == 0] * 1.).numpy()
                                        for a_s in act_sim])
        repBias_sims = np.array([np.mean(a_s[1:] == a_s[:-1]) for a_s in act_sim])

        # behavior of animal
        perf_1contrast = (trials_df.feedbackType.values > 0)[(trials_df.contrastRight == 1) + (trials_df.contrastLeft == 1)].mean()
        perf_0contrast = (trials_df.feedbackType.values > 0)[(trials_df.contrastRight == 0) + (trials_df.contrastLeft == 0)].mean()
        repBias = np.mean(trials_df.choice.values[1:] == trials_df.choice.values[:-1])

        # distance between both
        distance = (repBias_sims - repBias) ** 2 + (perf_0contrast_sims - perf_0contrast) ** 2 + (perf_1contrast_sims - perf_1contrast) ** 2

        act_sim = act_sim[np.argmin(distance)]

    return act_sim


def generate_imposter_session(imposterdf,
                              eid,
                              nbtrials,
                              nbSampledSess=10,
                              stitching_for_imposter_session=True,
                              pLeftChange_when_stitch=True):
    """

    Parameters
    ----------
    imposterd: all sessions concatenated in pandas dataframe (generated with pipelines/03_generate_imposter_df.py)
    eid: eid of session of interest
    trials_df: dataframe of trials of interest
    nbSampledSess: number of imposter sessions sampled to generate the final imposter session. NB: the length
    of the nbSampledSess stitched sessions must be greater than the session of interest, so typically choose a large
    number. If that condition is not verified a ValueError will be raised.
    Returns
    -------
    imposter session df

    """
    # this is to correct for when the eid is not part of the imposterdf eids
    # which is very possible when using imposter sessions from biaisChoice world.
    if np.any(imposterdf.eid == eid):
        raise ValueError(
            'The eid of the current session was found in the imposter session df which is impossible as'
            'you generate the imposter sessions from biasChoice world and decoding from ehysChoice world'
        )
    temp_trick = list(imposterdf[imposterdf.eid == eid].template_sess.unique())
    temp_trick.append(-1)
    template_sess_eid = temp_trick[0]

    if stitching_for_imposter_session:
        imposter_eids = np.random.choice(
            imposterdf[imposterdf.template_sess != template_sess_eid].eid.unique(),
            size=nbSampledSess,
            replace=False)
    else:
        imposterdf['sess_size'] = imposterdf['template_sess'].map(imposterdf['template_sess'].value_counts().to_dict())
        imposter_eids = np.random.choice(
            imposterdf[(imposterdf.template_sess != template_sess_eid) * (imposterdf['sess_size'] >= nbtrials)].eid.unique(),
            size=1,
            replace=False)
    sub_imposterdf = imposterdf[imposterdf.eid.isin(imposter_eids)].reset_index(drop=True)
    sub_imposterdf['row_id'] = sub_imposterdf.index
    sub_imposterdf['sorted_eids'] = sub_imposterdf.apply(
        lambda x: (np.argmax(imposter_eids == x['eid']) * sub_imposterdf.index.size + x.row_id),
        axis=1)
    identical_comp = np.argmax(sub_imposterdf.eid.values[None] == imposter_eids[:, None],
                               axis=0) * sub_imposterdf.index.size + sub_imposterdf.row_id

    if np.any(identical_comp.values != sub_imposterdf['sorted_eids'].values):
        raise ValueError('There is certainly a bug in the code. Sorry!')

    if np.any(sub_imposterdf['sorted_eids'].unique() != sub_imposterdf['sorted_eids']):
        raise ValueError('There is most probably a bug in the function')

    sub_imposterdf = sub_imposterdf.sort_values(by=['sorted_eids'])
    # seems to work better when starting the imposter session as the actual session, with an unbiased block
    sub_imposterdf = sub_imposterdf[(sub_imposterdf.probabilityLeft != 0.5) |
                                    (sub_imposterdf.eid == imposter_eids[0])].reset_index(
                                        drop=True)
    if pLeftChange_when_stitch and stitching_for_imposter_session:
        valid_imposter_eids, current_last_pLeft = [], 0
        for i, imposter_eid in enumerate(imposter_eids):
            #  get first pLeft
            first_pLeft = sub_imposterdf[(
                sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[0]

            if np.abs(first_pLeft - current_last_pLeft) < 1e-8:
                first_pLeft_idx = sub_imposterdf[sub_imposterdf.eid == imposter_eid].index[0]
                second_pLeft_idx = sub_imposterdf[(sub_imposterdf.eid == imposter_eid) & (
                    sub_imposterdf.probabilityLeft.values != first_pLeft)].index[0]
                sub_imposterdf = sub_imposterdf.drop(np.arange(first_pLeft_idx, second_pLeft_idx))
                first_pLeft = sub_imposterdf[(
                    sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[0]

            if np.abs(first_pLeft - current_last_pLeft) < 1e-8:
                raise ValueError('There is certainly a bug in the code. Sorry!')

            #  make it such that stitches correspond to pLeft changepoints
            if np.abs(first_pLeft - current_last_pLeft) > 1e-8:
                valid_imposter_eids.append(
                    imposter_eid)  # if first pLeft is different from current pLeft, accept sess
                #  take out the last block on the first session to stitch as it may not be a block with the right
                #  statistics (given the mouse stops the task there)
                second2last_pLeft = 1 - sub_imposterdf[
                    (sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[-1]
                second2last_block_idx = sub_imposterdf[(sub_imposterdf.eid == imposter_eid) & (
                    np.abs(sub_imposterdf.probabilityLeft - second2last_pLeft) < 1e-8)].index[-1]
                last_block_idx = sub_imposterdf[(sub_imposterdf.eid == imposter_eid)].index[-1]
                sub_imposterdf = sub_imposterdf.drop(
                    np.arange(second2last_block_idx + 1, last_block_idx + 1))
                #  update current last pLeft
                current_last_pLeft = sub_imposterdf[(
                    sub_imposterdf.eid == imposter_eid)].probabilityLeft.values[-1]
                if np.abs(second2last_pLeft - current_last_pLeft) > 1e-8:
                    raise ValueError('There is most certainly a bug here')
        sub_imposterdf = sub_imposterdf[sub_imposterdf.eid.isin(valid_imposter_eids)].sort_values(
            by=['sorted_eids'])
        if sub_imposterdf.index.size < nbtrials:
            raise ValueError(
                'you did not stitch enough imposter sessions. Simply increase the nbSampledSess argument'
            )
        sub_imposterdf = sub_imposterdf.reset_index(drop=True)
    # select a random first block index <- this doesn't seem to work well, it changes the block statistics too much
    # idx_chge = np.where(sub_imposterdf.probabilityLeft.values[1:] != sub_imposterdf.probabilityLeft.values[:-1])[0]+1
    # random_number = np.random.choice(idx_chge[idx_chge < (sub_imposterdf.index.size - trials_df.index.size)])
    # imposter_sess = sub_imposterdf.iloc[random_number:(random_number + trials_df.index.size)].reset_index(drop=True)
    imposter_sess = sub_imposterdf.iloc[:nbtrials].reset_index(drop=True)
    return imposter_sess

