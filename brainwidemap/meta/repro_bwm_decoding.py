import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from scipy.stats import kruskal, f_oneway
from statsmodels.stats.multitest import multipletests
from scipy.cluster import hierarchy

from one.api import ONE
from reproducible_ephys_functions import figure_style, labs
from dmn_bwm import get_allen_info


'''
Replottig BWM decoding results for the repro paper, grouped by labs,
testing for systematic lab biases
'''

# for vari plot
_, b, lab_cols = labs()

one = ONE()

dec_d = {'stimside': 'stimside', 'choice': 'choice',
        'feedback': 'feedback', 'wheel-speed': 'wheel-speed'}         
          
dec_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'decoding')         


def bwm_scores(nscores=3, tt='stripplot', sb='lab'):
    """
    Analyze decoding and encoding scores across regions grouped by labs or animals.
    
    Parameters:
    - nscores: Minimum number of scores for a lab/region to be included.
    - ana: Analysis type ('dec' or 'enc'), for decoding or encoding (GLM).
    - sb: Sort by 'lab' or 'animals'.
    """

    varis = ['choice', 'stimside', 'feedback', 'wheel-speed']
    regs = ['VISa/am', 'CA1', 'DG', 'LP', 'PO']

    # Use loaded data paths as in `pool_results_across_analyses`
    _, pa = get_allen_info()
    
    # Pooled data paths
    ana = 'dec'
    analysis_path = dec_pth

    ps = {}
    fig, axs = plt.subplots(nrows=1, ncols=len(varis), sharex=True, sharey=True, figsize=(10.88, 7.03))
    k = 0
    
    for vari in varis:
        # Load pooled data based on `pool_results_across_analyses`
        
        data_file = analysis_path / f'{dec_d[vari]}_stage2.pqt'
        d = pd.read_parquet(data_file)
        pths = one.eid2path(d['eid'].values)
        d['lab'] = [b[str(p).split('/')[5]] for p in pths]
        d['subject'] = [str(p).split('/')[7] for p in pths]        
        d['region'] = d['region'].replace(['VISa', 'VISam'], 'VISa/am')
        d = d.dropna(subset=['score', 'lab', 'region', 'subject'])
        
        # Plot logic
        if tt == 'mean_std':
            reg_stats = d.groupby('region')['score'].agg(
                mean_score=np.nanmean, std_score=np.nanstd, count_scores='count'
            ).reset_index()
            reg_stats = reg_stats[reg_stats['count_scores'] >= nscores]
            
            x = reg_stats['mean_score'].values
            y = reg_stats['std_score'].values
            regions = reg_stats['region'].values
            cols = [pa[region] for region in regions]
            sizes = reg_stats['count_scores'].values

            axs[k].scatter(x, y, color=cols, s=sizes if ana == 'dec' else sizes / 10)
            for i, reg in enumerate(regions):
                axs[k].annotate(f'  {reg}', (x[i], y[i]), fontsize=5, color=cols[i])
                
            axs[k].set_title(vari)
            axs[k].set_xlabel('mean')
            axs[k].set_ylabel('std')
        
        elif tt == 'stripplot':
            filtered_data = d[d['region'].isin(regs)]
            labs_counts = filtered_data.groupby([sb, 'region'])['score'].count().reset_index(name='score_count')
            valid_labs_regions = labs_counts[labs_counts['score_count'] >= nscores]
            filtered_data = pd.merge(filtered_data, valid_labs_regions[[sb, 'region']], on=[sb, 'region'])

            labss = np.unique(filtered_data[sb].values)
            palette = {lab: lab_cols[lab] for lab in labss} if sb == 'lab' else None
            
            sns.stripplot(x='score', y='region', hue=sb, palette=palette, data=filtered_data, jitter=True if sb == 'lab' else False, dodge=True, ax=axs[k], order=regs, size=3)
            for i, region in enumerate(regs):
                if i == len(regs) - 1:
                    continue
                axs[k].axhline(i + 0.5, color='grey', linestyle='--')
                                    
            axs[k].set_title(vari)
            if sb == 'lab':
                if k != 0:
                    axs[k].legend([], [], frameon=False)
                else:
                    axs[k].legend(loc='lower left', fontsize=9, bbox_to_anchor=(-0.55, 1.04), ncols=len(labss)).set_draggable(True)
                  
            # ANOVA
            labs = np.unique(d[sb].values)
            for reg in regs:
                scores_by_lab = [d[(d[sb] == lab) & (d['region'] == reg)]['score'].values for lab in labs]
                filtered_scores_by_lab = [lab_scores for lab_scores in scores_by_lab if lab_scores.size >= nscores]
                
                if len(filtered_scores_by_lab) < 2:
                    continue
                
                F, p = kruskal(*filtered_scores_by_lab)
                ps[f"{vari}_{reg}"] = p
                m = np.max(np.concatenate(scores_by_lab))

                weight = 'bold' if p < 0.05 else 'normal'
                if vari == 'wheel-speed':
                    x = 0.6

                else:
                    x = 0.1    
                axs[k].text(x, regs.index(reg), f'F={F:.2f}\np={p:.3f}', weight=weight, ha='left', va='center', fontsize=8)

        k += 1

    if tt == 'stripplot':
        p_values_list = list(ps.values())
        _, ps_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_by')
        corrected_p_values_dict = dict(zip(ps.keys(), ps_corrected))
        for key, value in corrected_p_values_dict.items():
            print(f"{key}: p-value = {value:.3f}")    

    fig.subplots_adjust(top=0.922, bottom=0.088, left=0.094, right=0.982, hspace=0.2, wspace=0.211)
