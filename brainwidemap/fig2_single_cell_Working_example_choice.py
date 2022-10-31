#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#from oneibl.one import ONE
from one.api import ONE
import brainbox.io.one as bbone

from math import *
import sys
import scipy.stats as scist
from os import path

import pandas as pd

from scipy.stats import rankdata



######



one = ONE()

from brainbox.population.decode import get_spike_counts_in_bins

from brainbox.io.one import SpikeSortingLoader

from ibllib.atlas import AllenAtlas

ba = AllenAtlas()


from brainbox.task.closed_loop import generate_pseudo_blocks


# In[2]:


def TwoNmannWhitneyUshuf(x,y,nShuf=10000):
    
    nA1=len(x)
    nB1=len(y)
    ################### x>y #####################
    t1=np.zeros((nShuf+1,nA1))
    
    t2=np.append(x,y,axis=0)
    t=rankdata(t2)
    
    t1[0,:]=t[range(nA1)]
    for i in range(nShuf):
        z1=np.random.choice(nA1+nB1, size=nA1, replace=False)
        t1[i+1,:]=t[z1]
        
    if nA1==1:
        numer1=t1[:,0]
    else: 
        numer1=np.sum(t1,axis=1)
        
    numer=numer1-nA1*(nA1+1)/2 
    
    
    ############### y>x ###########################  
    t3=np.zeros((nShuf+1,nB1))
    
    t4=np.append(y,x,axis=0)
    t5=rankdata(t4)
    
    t3[0,:]=t5[range(nB1)]
    for i in range(nShuf):
        z1=np.random.choice(nA1+nB1, size=nB1, replace=False)
        t3[i+1,:]=t5[z1]
        
    if nB1==1:
        numer3=t3[:,0]
    else: 
        numer3=np.sum(t3,axis=1)
        
    numer2=numer3-nB1*(nB1+1)/2       

    ######################################################
    numer_final=np.minimum(numer,numer2)
    
    return numer_final

########### p-value for block side  ############

def get_block(rate,c_L,c_R,block_label,choice_label):
    
    #nShuf=10000;
    nShuf=5000;
    
    num_neuron=len(rate[:,0])
    
    
    
    p=np.zeros(num_neuron)
    
    
    for i_neuron in range(num_neuron):
    
        spike_count=rate[i_neuron,:]
    
    
    
        ############ left choice=1 ############
        con1=np.logical_and(c_L>0,choice_label==1)
        con2=np.logical_and(block_label==0.8,con1)
        con3=np.logical_and(block_label==0.2,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x2=spike_count[index1[:,0]]
        y2=spike_count[index2[:,0]]
    
    
        nA2=len(x2)
        nB2=len(y2)
    
    
        numer2 = TwoNmannWhitneyUshuf(x2,y2,nShuf)
    
    
        ############ left choice=-1 ############
        con1=np.logical_and(c_L>0,choice_label==-1)
        con2=np.logical_and(block_label==0.8,con1)
        con3=np.logical_and(block_label==0.2,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x5=spike_count[index1[:,0]]
        y5=spike_count[index2[:,0]]
    
    
        nA5=len(x5)
        nB5=len(y5)
    
    
        numer5 = TwoNmannWhitneyUshuf(x5,y5,nShuf)
    
    
    
    
    
    
        ############ left, block=0.2, choice=1 ############
        con1=np.logical_and(c_R>0,choice_label==1)
        con2=np.logical_and(block_label==0.8,con1)
        con3=np.logical_and(block_label==0.2,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x3=spike_count[index1[:,0]]
        y3=spike_count[index2[:,0]]
    
    
        nA3=len(x3)
        nB3=len(y3)
    
    
        numer3 = TwoNmannWhitneyUshuf(x3,y3,nShuf)
    
    
        ############ left, block=0.2, choice=-1 ############
        con1=np.logical_and(c_R>0,choice_label==-1)
        con2=np.logical_and(block_label==0.8,con1)
        con3=np.logical_and(block_label==0.2,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x6=spike_count[index1[:,0]]
        y6=spike_count[index2[:,0]]
    
    
        nA6=len(x6)
        nB6=len(y6)
    
    
        numer6 = TwoNmannWhitneyUshuf(x6,y6,nShuf)
    
    
    
    
        nTotal=  numer2+numer3+numer5+numer6
   
        dTotal = nA2*nB2 +nA3*nB3+ nA5*nB5 +nA6*nB6    
 
   
   
        cp = nTotal/dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0]/(1+nShuf)

    
    return p


########### p-value for visual stimulus-side############


def get_stim(rate,c_L,c_R,block_label,choice_label,nShuf=5000):
    
    #nShuf=10000;
    #nShuf=5000;
    
    num_neuron=len(rate[:,0])
    
    
    
    p=np.zeros(num_neuron)
    
    
    for i_neuron in range(num_neuron):
    
        spike_count=rate[i_neuron,:]
    
    
    
        ############ block=0.8, choice=1 ############
        con1=np.logical_and(block_label==0.8,choice_label==1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x2=spike_count[index1[:,0]]
        y2=spike_count[index2[:,0]]
    
    
        nA2=len(x2)
        nB2=len(y2)
    
    
        numer2 = TwoNmannWhitneyUshuf(x2,y2,nShuf)
    
    
        ############ block=0.8, choice=-1 ############
        con1=np.logical_and(block_label==0.8,choice_label==-1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x5=spike_count[index1[:,0]]
        y5=spike_count[index2[:,0]]
    
    
        nA5=len(x5)
        nB5=len(y5)
    
    
        numer5 = TwoNmannWhitneyUshuf(x5,y5,nShuf)
    
    
    
    
    
    
        ############  block=0.2, choice=1 ############
        con1=np.logical_and(block_label==0.2,choice_label==1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x3=spike_count[index1[:,0]]
        y3=spike_count[index2[:,0]]
    
    
        nA3=len(x3)
        nB3=len(y3)
    
    
        numer3 = TwoNmannWhitneyUshuf(x3,y3,nShuf)
    
    
        ############  block=0.2, choice=-1 ############
        con1=np.logical_and(block_label==0.2,choice_label==-1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x6=spike_count[index1[:,0]]
        y6=spike_count[index2[:,0]]
    
    
        nA6=len(x6)
        nB6=len(y6)
    
    
        numer6 = TwoNmannWhitneyUshuf(x6,y6,nShuf)
    
    
    
    
        nTotal=  numer2+numer3+numer5+numer6
   
        dTotal = nA2*nB2 +nA3*nB3+ nA5*nB5 +nA6*nB6    
 
   
   
        cp = nTotal/dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0]/(1+nShuf)

    
    return p



########### p-value for choice side (condition-combined methods) ############


def get_choice(rate,c_L,c_R,block_label,choice_label,nShuf=5000):
    
    #nShuf=10000;
    #nShuf=5000;
    
    num_neuron=len(rate[:,0])
    
    
    
    p=np.zeros(num_neuron)
    
    
    for i_neuron in range(num_neuron):
    
        spike_count=rate[i_neuron,:]
    
    
    
        ############ block=0.8, stim=L ############
        con1=np.logical_and(block_label==0.8,c_L>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x2=spike_count[index1[:,0]]
        y2=spike_count[index2[:,0]]
    
    
        nA2=len(x2)
        nB2=len(y2)
    
    
        numer2 = TwoNmannWhitneyUshuf(x2,y2,nShuf)
    
    
        ############ block=0.8, stim=R ############
        con1=np.logical_and(block_label==0.8,c_R>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x5=spike_count[index1[:,0]]
        y5=spike_count[index2[:,0]]
    
    
        nA5=len(x5)
        nB5=len(y5)
    
    
        numer5 = TwoNmannWhitneyUshuf(x5,y5,nShuf)
    
    
    
    
    
    
        ############ block=0.2, stim=L ############
        con1=np.logical_and(block_label==0.2,c_L>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x3=spike_count[index1[:,0]]
        y3=spike_count[index2[:,0]]
    
    
        nA3=len(x3)
        nB3=len(y3)
    
    
        numer3 = TwoNmannWhitneyUshuf(x3,y3,nShuf)
    
    
        ############ block=0.2, stim=R ############
        con1=np.logical_and(block_label==0.2,c_R>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x6=spike_count[index1[:,0]]
        y6=spike_count[index2[:,0]]
    
    
        nA6=len(x6)
        nB6=len(y6)
    
    
        numer6 = TwoNmannWhitneyUshuf(x6,y6,nShuf)
    
    
    
    
        nTotal=  numer2+numer3+numer5+numer6
   
        dTotal = nA2*nB2 +nA3*nB3+ nA5*nB5 +nA6*nB6    
 
   
   
        cp = nTotal/dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0]/(1+nShuf)

    
    return p



################## Condition-combined test for indivdiual block, control time drift effect #####################


def Time_TwoNmannWhitneyUshuf(x,y,bx,by,nShuf):
    
    nx=len(x)
    ny=len(y)


################### x>y #####################
    t1=np.zeros((nShuf+1,nx))
    
    t2=np.append(x,y,axis=0)
    t=rankdata(t2)
 
    
    t1[0,:]=t[range(nx)]
    


    block_list=np.intersect1d(bx, by)


    for i_Shuf in range(nShuf):
        Final_index=np.zeros(nx+ny)
        Final_index[:]=range(nx+ny)
    #### generate random permutation sequence for individual block ####        
        for i_block in range(len(block_list)):
    
            bx_index=np.argwhere(bx==block_list[i_block])
            by_index=np.argwhere(by==block_list[i_block])
            temp_index=np.append(bx_index,by_index+len(bx))
    
            z1=np.random.choice(len(bx_index)+len(by_index), size=(len(bx_index)+len(by_index)), replace=False)
            z=temp_index[z1]
            Final_index[temp_index]=z
        
        Final_index=Final_index.astype(int)        
        t1[i_Shuf+1,:]=t[Final_index[range(nx)]]    

    if nx==1:
        numer1=t1[:,0]
    else: 
        numer1=np.sum(t1,axis=1)
        
    numer=numer1-nx*(nx+1)/2   
    
    
################### y>x #####################
    t3=np.zeros((nShuf+1,ny))
    
    t4=np.append(y,x,axis=0)
    t5=rankdata(t4)
 
    
    t3[0,:]=t5[range(ny)]
    


    block_list=np.intersect1d(by, bx)


    for i_Shuf in range(nShuf):
        Final_index=np.zeros(nx+ny)
        Final_index[:]=range(nx+ny)
    #### generate random permutation sequence for individual block ####        
        for i_block in range(len(block_list)):
    
            bx_index=np.argwhere(bx==block_list[i_block])
            by_index=np.argwhere(by==block_list[i_block])
            temp_index=np.append(by_index,bx_index+len(by))
    
            z1=np.random.choice(len(bx_index)+len(by_index), size=(len(bx_index)+len(by_index)), replace=False)
            z=temp_index[z1]
            Final_index[temp_index]=z
        
        Final_index=Final_index.astype(int)       
        t3[i_Shuf+1,:]=t5[Final_index[range(ny)]]    

    if ny==1:
        numer2=t3[:,0]
    else: 
        numer2=np.sum(t3,axis=1)
        
    numer3=numer2-ny*(ny+1)/2  
    


    ######################################################
    numer_final=np.minimum(numer,numer3)
    
    return numer_final





########### p-value for visual stimulus-side############


def get_stim_time_shuffle(rate,c_L,c_R,block_label,choice_label,nShuf=3000):
    
    #nShuf=10000;
   # nShuf=5000;
    
    num_neuron=len(rate[:,0])
    
    
    
    p=np.zeros(num_neuron)
    
    ######## get label of unique blocks #############
    
    s_block=np.zeros(len(block_label))

    for i in range(1,len(block_label)):
        s_block[i]=s_block[i-1]
        if abs(block_label[i]-block_label[i-1])>0:
            s_block[i]=s_block[i]+1

    ##################################################
    
    
    
    for i_neuron in range(num_neuron):
    
        spike_count=rate[i_neuron,:]
    
    
    
        ############ block=0.8, choice=1 ############
        con1=np.logical_and(block_label==0.8,choice_label==1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x2=spike_count[index1[:,0]]
        y2=spike_count[index2[:,0]]
        
        bx2=s_block[index1[:,0]]
        by2=s_block[index2[:,0]]
    
    
        nA2=len(x2)
        nB2=len(y2)
    
    
        numer2 = Time_TwoNmannWhitneyUshuf(x2,y2,bx2,by2,nShuf)
    
    
        ############ block=0.8, choice=-1 ############
        con1=np.logical_and(block_label==0.8,choice_label==-1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x5=spike_count[index1[:,0]]
        y5=spike_count[index2[:,0]]
        
        bx5=s_block[index1[:,0]]
        by5=s_block[index2[:,0]]
    
    
        nA5=len(x5)
        nB5=len(y5)
    
    
        numer5 = Time_TwoNmannWhitneyUshuf(x5,y5,bx5,by5,nShuf)
    
    
    
    
    
    
        ############  block=0.2, choice=1 ############
        con1=np.logical_and(block_label==0.2,choice_label==1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x3=spike_count[index1[:,0]]
        y3=spike_count[index2[:,0]]
        
        bx3=s_block[index1[:,0]]
        by3=s_block[index2[:,0]]
    
    
        nA3=len(x3)
        nB3=len(y3)
    
    
        numer3 = Time_TwoNmannWhitneyUshuf(x3,y3,bx3,by3,nShuf)
    
    
        ############  block=0.2, choice=-1 ############
        con1=np.logical_and(block_label==0.2,choice_label==-1)
        con2=np.logical_and(c_L>0,con1)
        con3=np.logical_and(c_R>0,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x6=spike_count[index1[:,0]]
        y6=spike_count[index2[:,0]]
        
        bx6=s_block[index1[:,0]]
        by6=s_block[index2[:,0]]
    
    
        nA6=len(x6)
        nB6=len(y6)
    
    
        numer6 = Time_TwoNmannWhitneyUshuf(x6,y6,bx6,by6,nShuf)
    
    
    
    
        nTotal=  numer2+numer3+numer5+numer6
   
        dTotal = nA2*nB2 +nA3*nB3+ nA5*nB5 +nA6*nB6    
 
   
   
        cp = nTotal/dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0]/(1+nShuf)

    
    return p





########### p-value for choice side############


def get_choice_time_shuffle(rate,c_L,c_R,block_label,choice_label,nShuf=3000):
    
    #nShuf=10000;
   # nShuf=5000;
    
    num_neuron=len(rate[:,0])
    
    
    
    p=np.zeros(num_neuron)
    
    ######## get label of unique blocks #############
    
    s_block=np.zeros(len(block_label))

    for i in range(1,len(block_label)):
        s_block[i]=s_block[i-1]
        if abs(block_label[i]-block_label[i-1])>0:
            s_block[i]=s_block[i]+1

    ##################################################
    
    
    
    for i_neuron in range(num_neuron):
    
        spike_count=rate[i_neuron,:]
    
    
    
        ############ block=0.8, stim=L ############
        con1=np.logical_and(block_label==0.8,c_L>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x2=spike_count[index1[:,0]]
        y2=spike_count[index2[:,0]]
        
        bx2=s_block[index1[:,0]]
        by2=s_block[index2[:,0]]
    
    
        nA2=len(x2)
        nB2=len(y2)
    
    
        numer2 = Time_TwoNmannWhitneyUshuf(x2,y2,bx2,by2,nShuf)
    
    
        ############ block=0.8, stim=R ############
        con1=np.logical_and(block_label==0.8,c_R>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
        

    
        x5=spike_count[index1[:,0]]
        y5=spike_count[index2[:,0]]
        
        bx5=s_block[index1[:,0]]
        by5=s_block[index2[:,0]]
    
    
        nA5=len(x5)
        nB5=len(y5)
    
    
        numer5 = Time_TwoNmannWhitneyUshuf(x5,y5,bx5,by5,nShuf)
    
    
    
    
    
    
        ############ block=0.2, stim=L ############
        con1=np.logical_and(block_label==0.2,c_L>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x3=spike_count[index1[:,0]]
        y3=spike_count[index2[:,0]]
        
        bx3=s_block[index1[:,0]]
        by3=s_block[index2[:,0]]
    
    
        nA3=len(x3)
        nB3=len(y3)
    
    
        numer3 = Time_TwoNmannWhitneyUshuf(x3,y3,bx3,by3,nShuf)
    
    
        ############ block=0.2, stim=R ############
        con1=np.logical_and(block_label==0.2,c_R>0)
        con2=np.logical_and(choice_label==1,con1)
        con3=np.logical_and(choice_label==-1,con1)
        index1= np.argwhere(con2)
        index2= np.argwhere(con3)
    
        x6=spike_count[index1[:,0]]
        y6=spike_count[index2[:,0]]
        
        bx6=s_block[index1[:,0]]
        by6=s_block[index2[:,0]]
    
    
        nA6=len(x6)
        nB6=len(y6)
    
    
        numer6 = Time_TwoNmannWhitneyUshuf(x6,y6,bx6,by6,nShuf)
    
    
    
    
        nTotal=  numer2+numer3+numer5+numer6
   
        dTotal = nA2*nB2 +nA3*nB3+ nA5*nB5 +nA6*nB6    
 
   
   
        cp = nTotal/dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0]/(1+nShuf)

    
    return p


# In[9]:


def BWM_choice_test(pid, eid, probe, TimeWindow=np.array([-0.1, 0.0])):

    
    
    
     
    # load spike data
    # spikes.times
    # spikes.clusters
    # clusters.brain location
    # list(clusters[probe])= ['mlapdv','brainLocationIds_ccf_2017','depths','brainLocationAcronyms_ccf_2017','metrics','channels','acronym','atlas_id','x','y','z']
    #spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
    
    
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)


    # load trial data

    collections = one.list_collections(eid, filename='_ibl_trials*')
    trials = one.load_object(eid, collection=collections[0],obj='trials')

    stim_on=trials.stimOn_times
    response=trials.response_times
    feedback=trials.feedback_times
    contrast_R=trials.contrastRight
    contrast_L=trials.contrastLeft
    choice=trials.choice
    block=trials.probabilityLeft
    first_move=trials.firstMovement_times
    
    # compute first movement time, if =NaN, replace it by response time.

    #goCueRTs = []
    #stimOnRTs = []
    #trials = TrialData(eid)
    #wheel  = WheelData(eid)
    #if wheel.data_error == False:
    #            wheel.calc_trialwise_wheel(trials.stimOn_times, trials.feedback_times)
    #            wheel.calc_movement_onset_times(trials.stimOn_times)
    #            first_move= wheel.first_movement_onset_times        
    nan_index=np.argwhere(np.isnan(first_move))
    first_move[nan_index[:,0]]=response[nan_index[:,0]]
    

                 


    num_neuron=len(np.unique(spikes['clusters']))
    num_trial =len(stim_on)



    ########## Time bins %%%%%%%%%%%%%%%%%%%%
    
    # TimeWindow[0]=-0.2
    # TimeWindow[1]= 1
    # BinSize=0.02
    
    #num_bin=np.floor((TimeWindow[1]-TimeWindow[0])/BinSize).astype(int)
    
   
    # 3 conditions R:[0.25 1], [0], L:[0.25 1]
    #firing_rate=np.zeros((num_neuron,num_bin,num_cond))
    
    
    # number of trials included
    
    


    
    
    
    
    
    # pre-move
    spike_rate=np.zeros((num_neuron,num_trial))


    ############ compute firing rate ###################
    
    
    
        
    T_1= TimeWindow[0]
    T_2= TimeWindow[1]
    

        


    raw_events=np.array([first_move+T_1, first_move+T_2]).T
    events=raw_events
    
   
            

    spike_count, cluster_id = get_spike_counts_in_bins(spikes['times'],spikes['clusters'],events)
        # firing_rate.shape=(num_neuron,num_bin,num_condition) 
        #count_number[i_bin]=   np.nansum(spike_count,axis=1)

            

    spike_rate = spike_count/(T_2-T_1)

        #num_trial_cond=len(events[:,0])
        # spike_count.shape(num_neuron,num_trial_cond)
        #  cluster_id.shape(num_neuron,1)
        

    # rate_1=(num_neuron,)
    #rate_1=np.nanmean(np.nanmean(firing_rate,axis=1),axis=1)
    #rate_1=(np.nanmean(firing_rate[:,:,0],axis=1)
    area_label_1=clusters['atlas_id'][cluster_id] 


     
    # only include units with firing rates > 1 Hz 
    #rate_threshold=1 
    
    #rate_good=np.argwhere(rate_1>rate_threshold)[:,0]
    
    # only include units pass single unit QC criterion
    ks2_id=np.zeros(0)
    for i in range(len(cluster_id)):
        if clusters['amp_median'][cluster_id[i]]>50/1000000:
            if clusters['slidingRP_viol'][cluster_id[i]]==1 and clusters['noise_cutoff'][cluster_id[i]]<20:
                ks2_id=np.append(ks2_id,[i])
    ks2_id=ks2_id.astype(int)        
        
    #included_units=np.intersect1d(rate_good, ks2_id)
    #included_units=rate_good
    included_units=ks2_id
    included_units=included_units.astype(int)
    
    rate=spike_rate[included_units,:]
    

    area_label=area_label_1[included_units]
    
    ############ return cluster id ########################
    QC_cluster_id=cluster_id[included_units]
    
    ############ compute p-value for block ###################
    
    ########## p1=(p-value for each neurons): Pre-move, condtion_combind_test (controll for temporal drift) #############
    
    

    p_1=get_choice_time_shuffle(rate,contrast_L,contrast_R,block,choice,3000)
    

    ########## p2=(p-value for each neurons): Pre-move, condtion_combind_test #############


    p_2=get_choice(rate,contrast_L,contrast_R,block,choice,3000)
    
    
    return p_1, p_2, area_label,  QC_cluster_id


# In[8]:


pid='3675290c-8134-4598-b924-83edb7940269'
eid='15f742e1-1043-45c9-9504-f1e8a53c1744'
probe='probe00'


# In[ ]:


### example session ###
p_1, p_2, area_label,  QC_cluster_id = BWM_choice_test(pid, eid, probe)

