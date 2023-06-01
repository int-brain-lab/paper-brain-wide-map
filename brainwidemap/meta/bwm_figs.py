import pandas as pd
import numpy as np
from pathlib import Path

from one.api import ONE
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from ibllib.atlas.flatmaps import plot_swanson_vector
from brainwidemap.manifold.state_space_bwm import (plot_traj_and_dist,
                                                   plot_all)

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from PIL import Image


#sns.set(font_scale=1.5)
#sns.set_style('ticks')

ba = AllenAtlas()
br = BrainRegions()
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
          
# get pooled results here
meta_pth = Path(one.cache_dir, 'meta')
meta_pth.mkdir(parents=True, exist_ok=True)          

dec_pth = Path(one.cache_dir, 'decoding')
dec_pth.mkdir(parents=True, exist_ok=True)  


def load_meta_results(variable):
    ''' 
    Load meta results for Swanson and table
    
    variable: ['stim', ' choice', 'fback', 'block']
    '''
    
    res = pd.read_pickle(meta_pth / f"{variable}.pkl")
    res = res.replace(True, 1)
    res = res.replace(False, 0)

    # ## Apply logarithm to GLM results
    res.glm_effect = np.log10(
                res.glm_effect.clip(lower=1e-5))

    # Reorder columns to match ordering in Figure
    res = res[['euclidean_latency',
                               'euclidean_effect',
                               'glm_effect',
                               'mannwhitney_effect',
                               'decoding_effect',
                               'decoding_significant',
                               'decoding_frac_significant',
                               'mannwhitney_significant',
                               'euclidean_significant']]
    return res


def get_cmap_(split):
    '''
    for each split, get a colormap defined by Yanliang,
    updated by Chris
    '''
    dc = {'stim': ["#EAF4B3","#D5E1A0", "#A3C968",
                    "#86AF40", "#517146","#33492E"],
          'choice': ["#F8E4AA","#F9D766","#E8AC22",
                     "#DA4727","#96371D"],
          'fback': ["#F1D3D0","#F5968A","#E34335",
                    "#A23535","#842A2A"],
          'block': ["#D0CDE4","#998DC3","#6159A6",
                    "#42328E", "#262054"]}

    if '_' in split:
        split = split.split('_')[0]

    return LinearSegmentedColormap.from_list("mycmap", dc[split])


def beryl_to_cosmos(beryl_acronym,br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    return br.get(ids=br.remap(beryl_id, source_map='Beryl', 
                  target_map='Cosmos'))['acronym'][0]


def swanson_to_beryl_hex(beryl_acronym,br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    rgb = br.get(ids=beryl_id)['rgb'][0].astype(int)
    return '#' + rgb_to_hex((rgb[0],rgb[1],rgb[2]))


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def plot_swansons(variable, fig=None, axs=None):

    res = load_meta_results(variable) 

    lw = .01
    
    # Labels for colorbars
    labels = ['Abs. diff. ' + r'$\Delta R^2$',
              'Latency (s)',
              'Nrml. Eucl. dist.',
              'Frac. sig. cells',
              'Decoding. ' + r'$R^2$' ' over null']

    # results to plot in Swansons
    res_types = ['glm_effect', 
                 'euclidean_latency',
                 'euclidean_effect',
                 'mannwhitney_effect',
                 'decoding_effect']

    # reverse list (until we can flip Swansons
    res_types.reverse()
    labels.reverse()
 
    cmap = get_cmap_(variable)
    if not fig:
        fig = plt.figure(figsize=(8,3))  
        gs = gridspec.GridSpec(len(res_types), 1, figure=fig,hspace=.75)
        axs = []
 
                 
    k = 0
    for res_type in res_types:
        if not fig:
            axs.append(fig.add_subplot(gs[0,k]))
        ana = res_type.split('_')[0]
        lat = True if (res_type.split('_')[1] == 'latency') else False
        

        if ana != 'glm':
            # check if there are p-values
            
            acronyms = res[res[f'{ana}_significant'] == True].index.values
            scores = res[res[f'{ana}_significant'] == True][res_types[k]].values
            mask = res[res[f'{ana}_significant'] == False].index.values
        
        else:
            acronyms = res.index.values
            scores = res[res_types[k]].values
            mask = None

        plot_swanson_vector(acronyms,
                            scores,
                            hemisphere=None, 
                            orientation='portrait',
                            cmap=cmap.reversed() if lat else cmap,
                            br=br,
                            ax=axs[k],
                            empty_color="white",
                            linewidth=lw,
                            mask=mask,
                            mask_color='silver',
                            annotate=True,
                            annotate_n=5,
                            annotate_order='bottom' if lat else 'top')
                            

        clevels = (min(scores), max(scores))
                   
        norm = matplotlib.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
        cbar = fig.colorbar(
                   matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),
                   ax=axs[k],shrink=0.75,aspect=12,pad=.025,
                   orientation="horizontal")
                   
        cbar.ax.tick_params(labelsize=5,rotation=90)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=1)
        cbar.ax.xaxis.set_tick_params(pad=5)
        cbar.set_label(labels[k], fontsize=6)
            
        axs[k].set_xticks([])
        axs[k].set_yticks([])
        axs[k].axis("off")
                        
        k += 1    


def plot_table(variable):


    # # Plot comparison table
    res = load_meta_results(variable)
    cmap = get_cmap_(variable)
    
    # ## Normalize values in each column to interval [0,1]

    res.iloc[:,:5] = (res.iloc[:,:5] - res.iloc[:,:5].min())/(
                      res.iloc[:,:5].max()-res.iloc[:,:5].min()) + 1e-4

    # ## Sum values in each row to use for sorting
    # The rows of the table are sorted by the sum of all effects 
    # across the row(excluding latency). 
    # Here we create a new column with this sum.

    res['sum']  = res[['decoding_effect',
                       'mannwhitney_effect',
                       'euclidean_effect',
                       'glm_effect']].apply(np.sum,axis=1)
                       
    res = res.reset_index()

    # ## Sort rows by 'sum' within each Cosmos region
    # The sorting by sum of effects is done within each Cosmos region. 
    # So here I add the cosmos acronym as a column, group and then sort 'sum'.

    res['cosmos'] = res.region.apply(lambda x : beryl_to_cosmos(x,br))
    res = res.groupby('cosmos').apply(
              lambda x: x.sort_values(['sum'], ascending=False))

    # ## Add hex values for Beryl regions
    # Here I add a column of hex values corresponding to beryl acronyms. 
    # This is used to color each row by region.

    res['beryl_hex'] = res.region.apply(swanson_to_beryl_hex,args=[br])    
    beryl_palette = dict(zip(res.region, res.beryl_hex))
    
    # Add dummy column to be colored according beryl rgb
    res['region_color'] = res.region 

    # ## Order columns according to panels in Figure

    res = res[['region','region_color', 
               'decoding_effect','mannwhitney_effect',
               'glm_effect','euclidean_effect',
               'decoding_significant',
               'mannwhitney_significant','euclidean_significant']]
            
    for rt in ['decoding', 'mannwhitney','euclidean']:
        res[f'{rt}_effect'] = res[f'{rt}_effect'] * res[f'{rt}_significant']
    
    res = res[['region','region_color',
               'glm_effect','euclidean_effect',
               'mannwhitney_effect','decoding_effect']]


    ## format table

    def region_formatting(x):
        '''
        Formatting for acronym strings
        '''
        color = beryl_palette[x]
        return 'background-color: ' + color


    def effect_formatting(x):
        '''
        Formatting for effect columns
        '''
        if x==0:
                color = 'silver'
        else:
            rgb = cmap(x)
            color =  ('#' + rgb_to_hex((int(255*rgb[0]),
                                        int(255*rgb[1]),
                                        int(255*rgb[2]))))
                         
        return 'background-color: ' + color


    def significance_formatting(x):
        if x==True:
            color = colors[-1]
            
        if x==False:
            color =  colors[0]
            
        if pd.isna(x):
            color='silver'
        return 'background-color: ' + color
        
        
    # Format table  
    def make_pretty(styler):
        
        styler.applymap(effect_formatting,
            subset=['decoding_effect','mannwhitney_effect',
                    'euclidean_effect','glm_effect'])
        styler.applymap(region_formatting,subset=['region_color'])
        styler.set_properties(subset=['region_color'], 
                              **{'width': '15px'})
        styler.set_properties(subset=['region_color'] , 
                              **{'font-size': '0pt'})
        styler.set_properties(subset=['decoding_effect',
                                      'euclidean_effect',
                                      'mannwhitney_effect',
                                      'glm_effect'], **{'width': '16px'})
        styler.set_properties(subset=['region'], **{'width': '65px'})
        styler.set_properties(subset=['decoding_effect',
                                      'euclidean_effect',
                                      'mannwhitney_effect',
                                      'glm_effect'] , **{'font-size': '0pt'})
        styler.set_properties(subset=['region'] , **{'font-size': '9pt'})
        styler.hide(axis="index")
        styler.set_table_styles([
            {"selector": "tr", "props": "line-height: 11px"},
            {"selector": "td,th", 
                "props": "line-height: inherit; padding: 0 "},
            {"selector": "tbody td", 
                "props": [("border", "1px solid white")]},
            {'selector': 'thead', 'props': [('display', 'none')]}])
            
        return styler


    ## Plot table
    res = res.style.pipe(make_pretty)
    res.export_png(f'Figures/{variable}_df_styled.png', 
                   max_rows=-1,
                   dpi = 200)

'''
#####
Brandon's decoding panels
#####
'''

def acronym2name(acronym):
    return br.name[np.argwhere(br.acronym==acronym)[0]][0]

def get_xy_vals(xy_table, eid, region):
    xy_vals = xy_table.loc[xy_table['eid_region']==f'{eid}_{region}']
    assert xy_vals.shape[0] == 1
    return xy_vals.iloc[0]

def get_res_vals(res_table, eid, region):
    er_vals = res_table[(res_table['eid']==eid) & (res_table['region']==region)]
    assert len(er_vals)==1
    return er_vals.iloc[0]


def stim_dec_line(fig=None, ax=None):

    '''
    plot decoding extra panels for bwm main figure
    '''

    if not fig: 
        fig, ax = plt.subplots(figsize=(3,2))
        
    variable = 'stim'
    eid = '5d01d14e-aced-4465-8f8e-9a1c674f62ec'
    region = 'VISp'

    file_all_results = dec_pth/f'{variable}.csv'
    file_xy_results = dec_pth/f'{variable}.pkl'
    extra_file = dec_pth/f'stim_{eid}_{region}.npy'

    res_table = pd.read_csv(file_all_results)
    xy_table = pd.read_pickle(file_xy_results)
    xy_vals = get_xy_vals(xy_table, eid, region)
    er_vals = get_res_vals(res_table, eid, region)

    l = xy_vals['regressors'].shape[0]
    X = np.squeeze(xy_vals['regressors']).T
    ws = np.squeeze(xy_vals['weights'])
    assert len(ws.shape) == 3
    W = np.stack([np.ndarray.flatten(ws[:,:,i]) 
                  for i in range(ws.shape[2])]).T
    assert W.shape[0] == 50
    mask = xy_vals['mask']
    preds = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
    targs = np.squeeze(xy_vals['targets'])
    trials = np.arange(len(mask))[[m==1 for m in mask]]
             
    targ_conts, trials_in = np.load(extra_file)
    assert np.all(trials == trials_in)
    u_conts = np.unique(targ_conts)
    neurometric_curve = 1-np.array([np.mean(preds[targ_conts==c]) 
                                    for c in u_conts])
    neurometric_curve_err = np.array([2*np.std(preds[targ_conts==c])
                                     /np.sqrt(np.sum(targ_conts==c)) 
                                      for c in u_conts])

    ax.set_title(f"{region}, single session ")                                  
    ax.plot(-u_conts, neurometric_curve, lw = 2, c='k')
#    ax.plot(-u_conts, neurometric_curve, 'ko', ms=8)
#    ax.errorbar(-u_conts, neurometric_curve, 
#                 neurometric_curve_err, color='k')
    ax.set_ylim(0,1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-1.03,1.03)
    ax.set_xticks([-1.0, -0.25, -0.125, -0.0625, 
                 0, 0.0625, 0.125, 0.25, 1.0])
                 
    ax.set_xticklabels([-1] + ['']*7 + [1])             
    ax.set_xlabel('Stimulus contrast')
    ax.set_ylabel('predicted \n P(stim = right)')

    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()


def dec_scatter(variable,fig=None, ax=None):

    '''
    plot decoding scatter for
    variable in [choice, fback, block]
    '''   

    scat_ex = {'choice': ['671c7ea7-6726-4fbe-adeb-f89c2c8e489b','GRN'], 
               'fback': ['e012d3e3-fdbc-4661-9ffa-5fa284e4e706','IRN'],
               'block': ['9e9c6fc0-4769-4d83-9ea4-b59a1230510e', 'MOp']}
               
    red = (255/255, 48/255, 23/255)
    blue = (34/255,77/255,169/255)
    
    if not fig: 
        fig, ax = plt.subplots(figsize=(3,2))        

    eid, region = scat_ex[variable]

    file_all_results = dec_pth/f'{variable}.csv'
    file_xy_results = dec_pth/f'{variable}.pkl'

    res_table = pd.read_csv(file_all_results)
    xy_table = pd.read_pickle(file_xy_results)
    xy_vals = get_xy_vals(xy_table, eid, region)
    er_vals = get_res_vals(res_table, eid, region)

    l = xy_vals['regressors'].shape[0]
    X = np.squeeze(xy_vals['regressors']).T
    ws = np.squeeze(xy_vals['weights'])
    assert len(ws.shape) == 3
    W = np.stack([np.ndarray.flatten(ws[:,:,i]) for i in range(ws.shape[2])]).T
    assert W.shape[0] == 50
    mask = xy_vals['mask']
    preds = np.mean(np.squeeze(xy_vals['predictions']), axis=0)
    targs = np.squeeze(xy_vals['targets'])
    trials = np.arange(len(mask))[[m==1 for m in mask]]

    ax.set_title(f"{region}, single session")
   
    ax.plot(trials[targs==0], 
            preds[targs==0] if variable=='fback' else 1-preds[targs==0],
             'o', c = red, lw=2,ms=2)
             
    ax.plot(trials[targs==1], 
            preds[targs==1] if variable=='fback' else 1-preds[targs==1],
             'o', c = blue, lw=2,ms=2)
    
    if variable == 'fback':
        l = ['Incorrect', 'Correct']
    elif variable == 'choice':
        l = ['Right choice', 'Left choice']
    elif variable == 'block':
        l = ['Right block', 'Left block']

    ax.legend(l,frameon=True)  
    ax.set_yticks([0, 0.5, 1])

    ax.set_xlim(100,400)

    ax.set_xlabel('Trials')
    target = l[0] if variable != 'fback' else l[1]
    ax.set_ylabel(f'Predicted \n {target}')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()

 
'''
###########
combine all panels
###########
'''
    

def main_fig(variable):

    '''
    combine panels into main figure
    using mosaic grid of 8 rows and 15 columns
    '''
    
    fig = plt.figure(figsize=(8, 13), facecolor='w')
    
    # Swansons
    s = ['glm_eff', 'euc_lat', 'euc_eff', 'man_eff', 'dec_eff']
    s2 = [[x]*3 for x in s]
    s3 = [[item for sublist in s2 for item in sublist]]*3

    # panels under swansons and table
    pe = [['p0', 'p0', 'p0', 'p0', 'p0', 
           'dec', 'dec', 'dec', 'dec', 'dec', 
           'tra_3d', 'tra_3d', 'tra_3d', 'tra_3d', 'tra_3d'],
          ['p3', 'p3', 'p3', 'p3', 'p3', 
           'ex_d', 'ex_d', 'ex_d', 'ex_d', 'ex_d',
           'tra_3d', 'tra_3d', 'tra_3d', 'tra_3d', 'tra_3d'],
          ['p5', 'p5', 'p5', 'p5', 'p5', 
           'ex_ds', 'ex_ds', 'ex_ds', 'ex_ds', 'ex_ds',
           'scat', 'scat', 'scat', 'scat', 'scat'],
          ['p8', 'p8', 'p8', 'p8', 'p8', 
           'ex_ds', 'ex_ds', 'ex_ds', 'ex_ds', 'ex_ds',
           'scat', 'scat', 'scat', 'scat', 'scat']]

    # 
    mosaic = s3 + [['tab']*15] + pe
        
    axs = fig.subplot_mosaic(mosaic)

    # 4 Swansons on top
    plot_swansons(variable, fig=fig, axs=[axs[x] for x in s])

    # plot table, reading from png
    if not Path(f'/home/mic/Figures/{variable}_df_styled.png').is_file():
        plot_table(variable)
    
    im = Image.open(f'/home/mic/Figures/{variable}_df_styled.png')
    axs['tab'].imshow(im.rotate(90, expand=True), aspect='auto')
                      
    axs['tab'].axis('off')                     
                      
    example_regs = {'stim': 'VISp', 
                    'choice': 'GRN', 
                    'fback': 'IRN', 
                    'block': 'PL'}
    
    # trajectory extra panels
    # example region 3d trajectory (tra_3d), line plot (ex_d)
    
    axs['tra_3d'].axis('off')
    axs['tra_3d'] = fig.add_subplot(4,2,6,projection='3d')
    axs['tra_3d'].axis('off')
    
    # manifold example region line plot
                   
    plot_traj_and_dist('stim', 'VISp', fig = fig,
                       axs=[axs['tra_3d'],axs['ex_d']])
    axs['tra_3d'].axis('off')

    # manifold panels, line plot with more regions and scatter    
    plot_all(splits=[variable], fig=fig, 
             axs=[axs['ex_ds'], axs['scat']]) 

    # decoding panel
    if variable == 'stim':
        stim_dec_line(fig=fig, ax=axs['dec'])
    else:   
        dec_scatter(variable,fig=fig, ax=axs['dec'])

    for k in axs:
        if k[0] == 'p':
            axs[k].axis('off')

    fig.subplots_adjust(top=0.985,
                        bottom=0.06,
                        left=0.018,
                        right=0.992,
                        hspace=1.0,
                        wspace=1.0)

