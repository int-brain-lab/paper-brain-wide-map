#!/usr/bin/env python
# coding: utf-8
import pdfkit
import pandas as pd
import numpy as np
from functools import reduce
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from matplotlib.colors import LinearSegmentedColormap
import dataframe_image as dfi
from ibllib.atlas.flatmaps import plot_swanson_vector
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
from scipy import stats
ba = AllenAtlas()
br = BrainRegions()
#get_ipython().run_line_magic('matplotlib', 'qt')
from IPython.display import HTML

def load_results(variable):
    ''' Load results
    
    variable: ['stim', ' choice', 'fback', 'block']
    '''
    
    res = pd.read_pickle(f"Results/{variable}.pkl")
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


def plot_swansons(variable):

    res = load_results(variable) 

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

 
    cmap = get_cmap_(variable)
    fig = plt.figure(figsize=(8,3))
    gs = gridspec.GridSpec(len(res_types), 1, figure=fig,hspace=.75)    
    axs = []
 
                 
    k = 0
    for res_type in res_types:
    
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
                            ax=axs[-1],
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
                   ax=axs[-1],shrink=0.75,aspect=12,pad=.025)
                   
        cbar.ax.tick_params(labelsize=5,rotation=90)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=1)
        cbar.ax.xaxis.set_tick_params(pad=5)
        cbar.set_label(labels[k], fontsize=6)
            
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        axs[-1].axis("off")
                        
        k += 1    


#    fig.savefig('Figures/' + variable + '_swansons.svg',
#        format='svg',
#        dpi=450,
#    )







def plot_table(variable):


    # # Plot comparison table
    res = load_results(variable)
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
    
    html = res.to_html()
      
    # write html to file
    text_file = open(f'Figures/{variable}_df_styled.html', "w")
    text_file.write(html)
    text_file.close()    
    pdfkit.from_file(f'Figures/{variable}_df_styled.html',
                     f'Figures/{variable}_df_styled.pdf')
      
    dfi.export(res, f'Figures/{variable}_df_styled.png',
               max_rows = -1,table_conversion="matplotlib")

