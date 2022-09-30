



if __name__=='__main__':
    import pickle
    import numpy as np
    psth_df = pickle.load(open('../../../tests/save_psth_df.pkl', 'rb'))
    psth_df = psth_df[['probabilityLeft', 'signedContrast', 'psth_stimOn_right_hem', 'psth_stimOn_left_hem', 'eid']]
    psth_df = psth_df.loc[psth_df.psth_stimOn_right_hem.apply(lambda x: len(x) == 40) * psth_df.psth_stimOn_left_hem.apply(lambda x: len(x) == 40)]
    contra_leftRegion = psth_df[(psth_df.probabilityLeft == 0.2)].groupby('eid').psth_stimOn_left_hem.mean()
    ipsi_leftRegion = psth_df[(psth_df.probabilityLeft == 0.8)].groupby('eid').psth_stimOn_left_hem.mean()
    contra_rightRegion = psth_df[(psth_df.probabilityLeft == 0.8)].groupby('eid').psth_stimOn_right_hem.mean()
    ipsi_rightRegion = psth_df[(psth_df.probabilityLeft == 0.2)].groupby('eid').psth_stimOn_right_hem.mean()

    ipsi = np.array([(v + p) * 0.5 for (v, p) in zip(ipsi_leftRegion.values, ipsi_rightRegion.values)])
    contra = np.array([(v + p) * 0.5 for (v, p) in zip(contra_leftRegion.values, contra_rightRegion.values)])

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(ipsi.mean(axis=0))
    plt.plot(contra.mean(axis=0))
    plt.show()
    plt.draw()

    from scipy.stats import ttest_1samp, wilcoxon
    from itertools import accumulate
    delta = ipsi - contra
    tvals, pvals = ttest_1samp(delta, 0.)

    significant_clusters = {}
    for tvalsign in [-1, 1]:
        signpvals = (pvals < 0.001) * (np.sign(tvals) == tvalsign)

        nb_consecutive = 8
        signpvals_cumsum = np.array(list(accumulate(signpvals * 1, lambda x, y: (x + y) * (y > 0))))
        signpvals_cumsum[:-1] = signpvals_cumsum[:-1] * (signpvals_cumsum[1:] == 0)
        xx = np.where((signpvals_cumsum >= nb_consecutive) * signpvals)[0]
        N_samples = 10000

        for xx_cluster in xx:
            cluster = xx_cluster - np.arange(signpvals_cumsum[xx_cluster])
            summed_tvals = tvals[cluster].sum(axis=0)
            tval = ttest_1samp(delta[:, cluster, None] * (2 * np.random.randint(2, size=(len(delta), 1, N_samples)) - 1), 0)[0].sum(axis=0)
            if ((summed_tvals < tval).mean() > 0.95 and tvalsign == -1) or ((summed_tvals < tval).mean() < 0.05 and tvalsign == 1):
                significant_clusters[tvalsign] = cluster

    # over_contrast
    contra_leftRegion_0contrast = psth_df[(psth_df.probabilityLeft == 0.2) * (psth_df.signedContrast.abs() == 1)].groupby('eid').psth_stimOn_left_hem.mean()
    ipsi_leftRegion_0contrast = psth_df[(psth_df.probabilityLeft == 0.8) * (psth_df.signedContrast.abs() == 1)].groupby('eid').psth_stimOn_left_hem.mean()
    contra_rightRegion_0contrast = psth_df[(psth_df.probabilityLeft == 0.8) * (psth_df.signedContrast.abs() == 1)].groupby('eid').psth_stimOn_right_hem.mean()
    ipsi_rightRegion_0contrast = psth_df[(psth_df.probabilityLeft == 0.2) * (psth_df.signedContrast.abs() == 1)].groupby('eid').psth_stimOn_right_hem.mean()

    ipsi_0contrast = np.array([(v + p) * 0.5 for (v, p) in zip(ipsi_leftRegion_0contrast.values, ipsi_rightRegion_0contrast.values)])
    contra_0contrast = np.array([(v + p) * 0.5 for (v, p) in zip(contra_leftRegion_0contrast.values, contra_rightRegion_0contrast.values)])

    delta_0contrast = ipsi_0contrast - contra_0contrast
    delta_0contrast[:, significant_clusters[1]]

    wilcoxon(delta_0contrast[:, significant_clusters[1][5:-2]].mean(axis=-1))


    # plot wrt contrast
    X, y = [], []
    for abs_contrast in [[0], [0.0625], [0.125], [0.25], [1]]:
        psth_df = pickle.load(open('../../../tests/save_psth_df.pkl', 'rb'))
        psth_df = psth_df[['probabilityLeft', 'signedContrast', 'psth_stimOn_right_hem', 'psth_stimOn_left_hem', 'eid']]
        psth_df = psth_df.loc[psth_df.psth_stimOn_right_hem.apply(lambda x: len(x) == 40) * psth_df.psth_stimOn_left_hem.apply(lambda x: len(x) == 40)]
        contra_leftRegion = psth_df[(psth_df.probabilityLeft == 0.2) * (psth_df.signedContrast.abs().isin(abs_contrast))].groupby('eid').psth_stimOn_left_hem.mean()
        ipsi_leftRegion = psth_df[(psth_df.probabilityLeft == 0.8) * (psth_df.signedContrast.abs().isin(abs_contrast))].groupby('eid').psth_stimOn_left_hem.mean()
        contra_rightRegion = psth_df[(psth_df.probabilityLeft == 0.8) * (psth_df.signedContrast.abs().isin(abs_contrast))].groupby('eid').psth_stimOn_right_hem.mean()
        ipsi_rightRegion = psth_df[(psth_df.probabilityLeft == 0.2) * (psth_df.signedContrast.abs().isin(abs_contrast))].groupby('eid').psth_stimOn_right_hem.mean()

        ipsi = np.array([(v + p) * 0.5 for (v, p) in zip(ipsi_leftRegion.values, ipsi_rightRegion.values)])
        contra = np.array([(v + p) * 0.5 for (v, p) in zip(contra_leftRegion.values, contra_rightRegion.values)])

        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(ipsi.mean(axis=0))
        plt.plot(contra.mean(axis=0))
        plt.title('abs_contrast: %f' % abs_contrast[0])
        X.append((ipsi - contra)[:, 12:18])
        y.append(abs_contrast * 52)
    plt.show()
    plt.draw()

xmean = np.array([x.mean() for x in X])
xmean = np.array([x.mean() for x in X])

plt.figure()
plt.plot(np.array([0, 0.0625, 0.125, 0.25, 1]), xmean)
plt.xlabel('abs contrast')
plt.ylabel('difference between psths pre-stim')
plt.draw()
plt.show()