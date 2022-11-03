import numpy as np
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from oneibl import one
from brainbox.io.one import load_spike_sorting
from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import seaborn as sns

one = one.ONE()


def plot_save_popactivity(eid, probe):
    spikes, _ = load_spike_sorting(eid, one=one, probe=probe)
    spkt = spikes[probe].times
    spkclu = spikes[probe].clusters
    binnedspikes, _, __ = bincount2D(spkt, spkclu, xbin=1.)
    binnedspikes = binnedspikes.T
    midpoint = binnedspikes.shape[0] // 2

    norms = np.linalg.norm(binnedspikes, axis=1)
    embeddings = PCA(n_components=2).fit_transform(binnedspikes)
    p_values = ttest_ind(binnedspikes[:midpoint], binnedspikes[midpoint:])[1]

    tcolors = sns.cubehelix_palette(binnedspikes.shape[0])
    plt.figure()
    plt.plot(np.arange(norms.shape[0]) * (1 / 60), norms)
    plt.xlabel('Minutes since session start')
    plt.ylabel('Population norm')
    plt.title(eid + probe)
    plt.savefig('/media/berk/Storage1/fits/populationplots/' + eid + '_norm.png')
    plt.close()

    plt.figure()
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=tcolors)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(eid + probe)
    plt.savefig('/media/berk/Storage1/fits/populationplots/' + eid + '_pca.png')
    plt.close()

    plt.figure()
    plt.hist(p_values, bins=50)
    plt.title('Histogram of two-sample t-test p-values for ' + eid)
    plt.xlabel('p-value')
    plt.savefig('/media/berk/Storage1/fits/populationplots/' + eid + '_pvals.png')
    plt.close()
    return


if __name__ == '__main__':
    from ibl_pipeline import subject, ephys, histology
    from ibl_pipeline.analyses import behavior as behavior_ana
    regionlabeled = histology.ProbeTrajectory &\
        'insertion_data_source = "Ephys aligned histology track"'
    sessions = subject.Subject * subject.SubjectProject * ephys.acquisition.Session *\
        regionlabeled * behavior_ana.SessionTrainingStatus
    bwm_sess = sessions & 'subject_project = "ibl_neuropixel_brainwide_01"' &\
        'good_enough_for_brainwide_map = 1'
    sessinfo = [info for info in bwm_sess]

    for s in sessinfo:
        plot_save_popactivity(str(s['session_uuid']), 'probe0' + str(s['probe_idx']))
