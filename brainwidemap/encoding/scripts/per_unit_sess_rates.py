from one.api import ONE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from neurencoding.linear import LinearGLM
import neurencoding.utils as mut
import sklearn.preprocessing as pp


def session_rateplot(eid, probe=0, n_tails=40):
    trialsdf = bbone.load_trials_df(eid, maxlen=2., t_before=0.6, t_after=0.6)
    probestr = 'probe0' + str(probe)
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(eid, aligned=True)
    spk_times = spikes[probestr].times
    spk_clu = spikes[probestr].clusters
    # clu_regions = clusters[probestr].acronym
    
    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'pLeft_last': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'adj_contrastLeft': 'value',
                'contrastRight': 'value',
                'adj_contrastRight': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'bias': 'value',
                'bias_next': 'value',
                'wheel_velocity': 'continuous'}
    
    linglm = LinearGLM(trialsdf, spk_times, spk_clu, vartypes, binwidth=0.02)
    bases = mut.full_rcos(0.6, 10, linglm.binf)
    linglm.add_covariate_timing('stimon', 'stimOn_times', bases)
    linglm.compile_design_matrix()
    binnedspikes = linglm.binnedspikes
    normspikes = pp.minmax_scale(binnedspikes)
    spikesdf = pd.DataFrame(normspikes)
    spikesdf['trial'] = linglm.trlabels
    trialmeans = spikesdf.groupby('trial').agg('mean')
    trialmeans.columns.name = 'unit'
    first_tr_mean = trialmeans.iloc[:n_tails].mean()
    last_tr_mean = trialmeans.iloc[-n_tails:].mean()
    meandiffs = (last_tr_mean - first_tr_mean).sort_values(ascending=False)
    sortinds = meandiffs.index
    fig = plt.figure(figsize=(6, 10))
    ax = sns.heatmap(trialmeans.T.reindex(sortinds), cmap='gray_r')
    ax.set_title(f'Session {eid} mean per-trial activity\nsorted by change in mean')
    ax.set_ylabel('Unit')
    plt.tight_layout()
    return ax

if __name__ == '__main__':
    from bbglm_sessfit_linear_comp import get_bwm_ins_alyx

    one = ONE()
    ins, ins_ids, sess_ids = get_bwm_ins_alyx(one=one)

    for eid in sess_ids:
        try:
            ax = session_rateplot(eid)
        except Exception as e:
            print(e)
            continue
        plt.savefig(f'/home/berk/Documents/psth-plots/sessrates/{eid}_probe0.png', dpi=300)
        plt.close()

