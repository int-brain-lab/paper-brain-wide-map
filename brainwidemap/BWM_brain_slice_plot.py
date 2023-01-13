#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import brainbox.io.one as bbone

from math import *
import sys
import scipy.stats as scist
from os import path
import matplotlib.pyplot as plt


###############################
import numpy as np
from one.api import ONE

one = ONE()

from ibllib.atlas import AllenAtlas

# atlas of 10um resolution #
ba = AllenAtlas(10)

################# input: list_region_id, list_region_value ###############################
#################  list_region_id: numpy array of Beryl region id ########################
#################  list_region_value: numpy array of values, ordered by list_region_id ###
################# color_range=0:   
#################   default: if  0<value<0.05    RGB= S_color[0,:]
#################   default: if  0.05<value<0.1  RGB= S_color[1,:]
#################   default: if  0.1<value<0.15  RGB= S_color[2,:]
#################   default: if  0.15<value      RGB= S_color[3,:]


################# color_range=1:   d=(max-min)/4
#################    if  min<value<min+d        RGB= S_color[0,:]
#################    if  min+d<value<min+2d     RGB= S_color[1,:]
#################    if  min+2d<value<mind+3d    RGB= S_color[2,:]
#################    if  min+3d<value            RGB= S_color[3,:]

################ Define custom colors by changing S_color







######## create RGB image of sag view slice [pixel_x,pixel_y,3] ########
def sag_slice_RGB(list_region_id, list_region_value,ML_coordinate,color_range=0):

    
    #coordinate_1=(-4000+100*21)/1000000
    #coordinate_2=(-4000+100*31)/1000000
    #coordinate_3=(-4000+100*37)/1000000
    
    coordinate=ML_coordinate/1000000
    index_1 = ba.bc.x2i(np.array(coordinate), mode='raise')
    axis=0
    
    mode='raise'
    mapping='Beryl'
    sag_slice_ind=_take_remap(ba.label, index_1, ba.xyz2dims[axis], mapping)
    sag_slice_b=ba.slice( coordinate, axis=0, volume='boundary', mode='raise', region_values=None, mapping='Beryl', bc=None)

    #### 2D slices 
    sag_slice=np.transpose(sag_slice_ind)
    #### 2D slice boundary
    sag_slice_b=np.transpose(sag_slice_b)


    ########## color of values ##############
    S_color=np.array([ [249/255, 228/255, 183/255],[0.9882,0.8510,0.5020],[0.9294,0.6902,0.1294], [0.8510,0.3294,0.1020]])
    
    ##### color of Null vlaues white=[1,1,1]
    N_color=[1,1,1]
    
    #### set color range ######
    # set the color of regions
    # Local_value in [Range_0, Range_1]: color=S_color[0,:]
    # Local_value in [Range_1, Range_2]: color=S_color[1,:]
    # Local_value in [Range_2, Range_3]: color=S_color[2,:]
    # Local_value > Range_3: color=S_color[3,:]
    ## default color range ##
    if color_range==0:
        Range_0=0
        Range_1=0.05
        Range_2=0.1
        Range_3=0.15
           
    ## range is determined by the max and min value ##     
    elif color_range==1:
        
        d_value=(np.max(list_region_value)-np.min(list_region_value))/4
        
        Range_0=np.min(list_region_value)
        Range_1=np.min(list_region_value)+d_value
        Range_2=np.min(list_region_value)+2*d_value
        Range_3=np.min(list_region_value)+3*d_value
        
    
    
    

    #initial color of image: white=[1,1,1]
    im_sag_1=np.ones((len(sag_slice[:,0]),len(sag_slice[0,:]),3))

    
    for i_reg in range(len(list_region_id)):
    
        Local_value=list_region_value[i_reg]
        Local_region_id=list_region_id[i_reg]
    

        # set the color of regions
        # Local_value in [Range_0, Range_1]: color=S_color[0,:]
        # Local_value in [Range_1, Range_2]: color=S_color[1,:]
        # Local_value in [Range_2, Range_3]: color=S_color[2,:]
        # Local_value > Range_3: color=S_color[3,:]
        
        if Local_value<=Range_0:
            Local_color=N_color
        elif Local_value<Range_1:
            Local_color=S_color[0,:]
        elif Local_value<Range_2:
            Local_color=S_color[1,:]
        elif Local_value<Range_3:
            Local_color=S_color[2,:]
        else:
            Local_color=S_color[3,:]
            

    
        Local_region_index=np.argwhere(ba.regions.id==Local_region_id)[:,0]
        region_pixel=np.argwhere(sag_slice==Local_region_index)


        im_sag_1[region_pixel[:,0],region_pixel[:,1],0]=Local_color[0]
        im_sag_1[region_pixel[:,0],region_pixel[:,1],1]=Local_color[1]
        im_sag_1[region_pixel[:,0],region_pixel[:,1],2]=Local_color[2]



    # set boundary color black=[0,0,0]
    Boundary_color=[0,0,0]
    # input boundary of 2D slices
    temp_sag_b_1=sag_slice_b

    z_1=np.argwhere(temp_sag_b_1==1)

    # RGB boundary color of slice 
    im_sag_1[z_1[:,0],z_1[:,1],0]=Boundary_color[0]
    im_sag_1[z_1[:,0],z_1[:,1],1]=Boundary_color[0]
    im_sag_1[z_1[:,0],z_1[:,1],2]=Boundary_color[0]
    
    return im_sag_1




######## create RGB image of cortex top view slice [pixel_x,pixel_y,3] ########
def ctx_slice_RGB(list_region_id, list_region_value,color_range=0):

    
    #### 2D cortex top view slice  
    ix, iy = np.meshgrid(np.arange(ba.bc.nx), np.arange(ba.bc.ny))
    iz = ba.bc.z2i(ba.top)
    inds = ba._lookup_inds(np.stack((ix, iy, iz), axis=-1))
    ctx_slice = ba._get_mapping(mapping='Beryl')[ba.label.flat[inds]]

    #### 2D slice boundary
    ctx_slice_b=ctx_b = ba.compute_boundaries(ctx_slice)


    ########## color of values ##############
    S_color=np.array([ [249/255, 228/255, 183/255],[0.9882,0.8510,0.5020],[0.9294,0.6902,0.1294], [0.8510,0.3294,0.1020]])
    
    ##### color of Null vlaues white=[1,1,1]
    N_color=[1,1,1]
    
    #### set color range ######
    ## default color range ##
    if color_range==0:
        Range_0=0
        Range_1=0.05
        Range_2=0.1
        Range_3=0.15
           
    ## range is determined by the max and min value ##     
    elif color_range==1:
        
        d_value=(np.max(list_region_value)-np.min(list_region_value))/4
        
        Range_0=np.min(list_region_value)
        Range_1=np.min(list_region_value)+d_value
        Range_2=np.min(list_region_value)+2*d_value
        Range_3=np.min(list_region_value)+3*d_value
        
    
    
    

    #initial color of image: white=[1,1,1]
    im_ctx_1=np.ones((len(ctx_slice[:,0]),len(ctx_slice[0,:]),3))

    
    for i_reg in range(len(list_region_id)):
    
        Local_value=list_region_value[i_reg]
        Local_region_id=list_region_id[i_reg]
    

        # set the color of regions
        if Local_value<=Range_0:
            Local_color=N_color
        elif Local_value<Range_1:
            Local_color=S_color[0,:]
        elif Local_value<Range_2:
            Local_color=S_color[1,:]
        elif Local_value<Range_3:
            Local_color=S_color[2,:]
        else:
            Local_color=S_color[3,:]
            

    
        Local_region_index=np.argwhere(ba.regions.id==Local_region_id)[:,0]
        region_pixel=np.argwhere(ctx_slice==Local_region_index)


        im_ctx_1[region_pixel[:,0],region_pixel[:,1],0]=Local_color[0]
        im_ctx_1[region_pixel[:,0],region_pixel[:,1],1]=Local_color[1]
        im_ctx_1[region_pixel[:,0],region_pixel[:,1],2]=Local_color[2]



    # set boundary color black=[0,0,0]
    Boundary_color=[0,0,0]
    # input boundary of 2D slices
    temp_ctx_b_1=ctx_slice_b

    z_1=np.argwhere(temp_ctx_b_1==1)

    # RGB boundary color of slice 
    im_ctx_1[z_1[:,0],z_1[:,1],0]=Boundary_color[0]
    im_ctx_1[z_1[:,0],z_1[:,1],1]=Boundary_color[0]
    im_ctx_1[z_1[:,0],z_1[:,1],2]=Boundary_color[0]
    
    return im_ctx_1





def make_sag_plot(im_1,im_2,im_3):
    fig, axs = plt.subplots(1, 3,figsize=(20,6))
    axs[0].imshow(im_1)
    plt.sca(axs[0])
    ax = plt.gca()
    #hide xy-axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    axs[1].imshow(im_2)
    plt.sca(axs[1])
    ax = plt.gca()
    #hide xy-axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    axs[2].imshow(im_3)
    plt.sca(axs[2])
    ax = plt.gca()
    #hide xy-axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
    
def make_ctx_plot(im_1):  
    plt.imshow(im_1[:,0:round(len(im_1[0,:,0])/2),:])
    ax3=plt.gca()
    #hide xy-axis
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    


###### ML coordinate ##########
coord_1=(-4000+100*21)
coord_2=(-4000+100*31)
coord_3=(-4000+100*37)

#### generate sag slices 
im_sag_1=sag_slice_RGB(list_region_id, list_region_value,coord_1,1)
im_sag_2=sag_slice_RGB(list_region_id, list_region_value,coord_2,1)
im_sag_3=sag_slice_RGB(list_region_id, list_region_value,coord_3,1)


#### generate cortex top view slice 
im_ctx_1=ctx_slice_RGB(list_region_id, list_region_value,1)



make_sag_plot(im_sag_1,im_sag_2,im_sag_3)


# In[ ]:


make_ctx_plot(im_ctx_1)

