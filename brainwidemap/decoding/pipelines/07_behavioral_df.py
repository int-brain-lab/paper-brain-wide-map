import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
import utils
import pandas as pd
from one.api import ONE, One
from functions.neurometric import fit_get_shift_range
from braindelphi.decoding.functions.process_targets import optimal_Bayesian
import models.utils as mut
from scipy.stats import pearsonr, spearmanr, wilcoxon
import brainbox.io.one as bbone

one = One()
insdf = pd.read_parquet('neural/2022-02-05_decode_signcont_task_Lasso_align_goCue_times_100_pseudosessions_regionWise_timeWindow_-0_6_-0_1_neurometricPLeft_optimal_bayesian_pseudoSessions_unmergedProbes.parquet')
eids = insdf.index.get_level_values(1).unique()
subjects = insdf.index.get_level_values(0).unique()
MIN_RT = 0.08

outdict = {}
nb_simul_beh_shift = 10000


oracle_pLefts = []
priors_pLefts = []
#  session level
for eid in eids:
    print(eid)
    try:
        data = utils.load_session(eid, one=one)
    except:
        continue
    uniq_contrasts = np.array([-1., -0.25, -0.125, -0.0625, 0.,  0.0625,  0.125, 0.25,  1.])

    side, stim, act, oracle_pLeft = mut.format_data(data)
    prior = optimal_Bayesian(act, stim, side).numpy()
    oracle_pLefts.append(oracle_pLeft)
    priors_pLefts.append(prior)

    #trials_df = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])
    #reaction_times = (trials_df['firstMovement_times'] - trials_df['goCue_times']).values

    # take out negative reaction times and when the mouse doesn't perform an action
    stim = stim[(act != 0)]
    side = side[(act != 0)]
    prior = prior[(act != 0)]
    act = act[(act != 0)]

    pLeft_constrast = {c: np.mean(act[stim == c] == 1) for c in uniq_contrasts}
    no_integration_act = np.vstack([2 * np.random.binomial(1, pLeft_constrast[c], size=nb_simul_beh_shift) - 1 for
                                    c in stim])

    perfat0 = (act == side)[stim == 0].mean()
    t, p = wilcoxon((act == side)[stim == 0] * 1 - 0.5, alternative='greater')

    p_nointegration = np.mean((no_integration_act == side[:, None]).mean(axis=0) < (act == side).mean())

    p_0cont_nointegration = np.mean((no_integration_act[stim == 0] == side[stim == 0, None]).mean(axis=0)
                                    < (act == side)[stim == 0].mean())

    low_prob_idx_trials = [(prior < 0.5) * (stim == c) for c in uniq_contrasts]
    lowprob_arr = [uniq_contrasts,
                   [len(act[idx]) for idx in low_prob_idx_trials],
                   [(act[idx] == 1).mean() for idx in low_prob_idx_trials]]
    high_prob_idx_trials = [(prior > 0.5) * (stim == c) for c in uniq_contrasts]
    highprob_arr = [uniq_contrasts,
                    [len(act[idx]) for idx in high_prob_idx_trials],
                    [(act[idx] == 1).mean() for idx in high_prob_idx_trials]]

    full_neurometric = fit_get_shift_range([lowprob_arr, highprob_arr], False, nfits=500)

    outdict[eid] = [t, p, p_nointegration, p_0cont_nointegration, perfat0, full_neurometric['shift']]

behdf = pd.DataFrame.from_dict(outdict, orient='index', columns=['tvalue', 'pvalue', 'p_nointegration',
                                                                 'p_0cont_nointegration',
                                                                 '0contrast_perf', 'psychometric_shift'])
behdf.to_parquet('behavioral/beh_shift_session_0_5.parquet')

print(spearmanr(behdf['0contrast_perf'].values, behdf['psychometric_shift'].values))


one = One()
insdf = pd.read_parquet('neural/2022-02-05_decode_signcont_task_Lasso_align_goCue_times_100_pseudosessions_regionWise_timeWindow_-0_6_-0_1_neurometricPLeft_optimal_bayesian_pseudoSessions_unmergedProbes.parquet')
eids = insdf.index.get_level_values(1).unique()
subjects = insdf.index.get_level_values(0).unique()
outdict = {}
#  subject level
for subject in subjects:
    print(subject)
    subject_eids = insdf.xs(subject, level=0).index.get_level_values(0).unique()
    stimuli_arr, actions_arr, stim_sides_arr, pLeft_arr = [], [], [], []
    for eid in subject_eids:
        try:
            data = utils.load_session(eid, one=one)
        except:
            continue
        if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
            stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
            prior = optimal_Bayesian(actions, stimuli, stim_side).numpy()
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            pLeft_arr.append(prior)
    uniq_contrasts = np.array([-1., -0.25, -0.125, -0.0625, 0.,  0.0625,  0.125, 0.25,  1.])

    # format data
    stim, act, side, prior = utils.format_input(stimuli_arr, actions_arr, stim_sides_arr, pLeft_arr)

    # take out negative reaction times and when the mouse doesn't perform an action
    stim = stim[(act != 0)]
    side = side[(act != 0)]
    prior = prior[(act != 0)]
    act = act[(act != 0)]

    pLeft_constrast = {c: np.mean(act[stim == c] == 1) for c in uniq_contrasts}
    no_integration_act = np.vstack([2 * np.random.binomial(1, pLeft_constrast[c], size=nb_simul_beh_shift) - 1 for
                                    c in stim])

    perfat0 = (act == side)[stim == 0].mean()
    t, p = wilcoxon((act == side)[stim == 0] * 1 - 0.5, alternative='greater')

    p_nointegration = np.mean((no_integration_act == side[:, None]).mean(axis=0) < (act == side).mean())

    p_0cont_nointegration = np.mean((no_integration_act[stim == 0] == side[stim == 0, None]).mean(axis=0)
                                    < (act == side)[stim == 0].mean())

    low_prob_idx_trials = [(prior < 0.3) * (stim == c) for c in uniq_contrasts]
    lowprob_arr = [uniq_contrasts,
                   [len(act[idx]) for idx in low_prob_idx_trials],
                   [(act[idx] == 1).mean() for idx in low_prob_idx_trials]]
    high_prob_idx_trials = [(prior > 0.7) * (stim == c) for c in uniq_contrasts]
    highprob_arr = [uniq_contrasts,
                    [len(act[idx]) for idx in high_prob_idx_trials],
                    [(act[idx] == 1).mean() for idx in high_prob_idx_trials]]

    full_neurometric = fit_get_shift_range([lowprob_arr, highprob_arr], False, nfits=500)

    outdict[subject] = [t, p, p_nointegration, p_0cont_nointegration, perfat0, full_neurometric['shift']]

behdf = pd.DataFrame.from_dict(outdict, orient='index', columns=['tvalue', 'pvalue', 'p_nointegration',
                                                                 'p_0cont_nointegration',
                                                                 '0contrast_perf', 'psychometric_shift'])
behdf.to_parquet('behavioral/beh_shift_subject.parquet')


# plot
len(oracle_pLefts)
print(pearsonr(np.concatenate(oracle_pLefts), np.concatenate(priors_pLefts)))

around_reversals = []
for i, oracle_pLeft in enumerate(oracle_pLefts):
    idxs_rev = np.where(oracle_pLeft[1:] != oracle_pLeft[:-1])[0] + 1
    for idx_rev in idxs_rev:
        if idx_rev < (len(oracle_pLefts) - 20) and oracle_pLeft[idx_rev - 1] != 0.5:
            if oracle_pLeft[idx_rev] == 0.8:
                around_reversals.append(priors_pLefts[i][(idx_rev - 5):(idx_rev + 20)])
            elif oracle_pLeft[idx_rev] == 0.2 and False:
                around_reversals.append(1 - priors_pLefts[i][(idx_rev - 5):(idx_rev + 20)])

around_reversals = np.array(around_reversals)
around_reversals.mean(axis=0)

plt.figure()
plt.plot(np.arange(-5, 20), around_reversals.mean(axis=0))
plt.fill_between(np.arange(-5, 20), around_reversals.mean(axis=0) - around_reversals.std(axis=0),
                 around_reversals.mean(axis=0) + around_reversals.std(axis=0), alpha=0.5)
plt.savefig('rev_opt_Bay.pdf')