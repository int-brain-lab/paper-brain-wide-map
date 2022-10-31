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

import matplotlib.pyplot as plt

######



one = ONE()

from brainbox.population.decode import get_spike_counts_in_bins

from brainbox.io.one import SpikeSortingLoader

from ibllib.atlas import AllenAtlas

ba = AllenAtlas()


# In[2]:


################## Condition-combined test for indivdiual block, control time drift effect #####################

######## Two-side Mann Whiteny U test ############
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





########### p-value for single-cell correlates with visual stimulus-side ############


def get_stim_time_shuffle(rate,c_L,c_R,block_label,choice_label,nShuf=3000):
    # rate=spike_count(num_neuron, num_trial)
    # c_L=contrast_L(num_trial)
    # c_R=contrast_R(num_trial)
    # block_label (num_trial)  [0.8,0.2]
    # choice_label=(num_trial) [1,-1]
    # nShuf= number of shuffle for null distribution
    # output: p=p-value for stim side(num_neuron)

    
    
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
    
    
    
        ########### combine all conditions ################
        nTotal=  numer2+numer3+numer5+numer6
   
        dTotal = nA2*nB2 +nA3*nB3+ nA5*nB5 +nA6*nB6    
 
   
        ######## compute discrimination probability and p-value for each cell ###########
        cp = nTotal/dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0]/(1+nShuf)

    
    return p


# In[8]:


def single_trial_PSTH(eid, probe, TimeWindow=np.array([-0.4, 0.3]), BinSize=0.01):

    
    
    
     
    # load spike data
    # spikes.times
    # spikes.clusters
    # clusters.brain location
    # list(clusters[probe])= ['mlapdv','brainLocationIds_ccf_2017','depths','brainLocationAcronyms_ccf_2017','metrics','channels','acronym','atlas_id','x','y','z']
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
    


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
    

                 


    num_neuron=len(np.unique(spikes[probe]['clusters']))
    num_trial =len(stim_on)




    # compute correct choice of trial (include correct trials or trials with zero contrast)
    left_stim_index=np.argwhere(contrast_L>0)
    right_stim_index=np.argwhere(contrast_R>0)

    trial_answer=np.zeros(len(choice))
    trial_answer[left_stim_index[:,0]]=1
    trial_answer[right_stim_index[:,0]]=-1

    
    
    


    
    
    
    num_bin=np.floor((TimeWindow[1]-TimeWindow[0])/BinSize).astype(int)
    
    
    spike_rate=np.zeros((num_neuron,num_trial,num_bin))
    spike_rate_2=np.zeros((num_neuron,num_trial,num_bin))
    spike_rate_3=np.zeros((num_neuron,num_trial,num_bin))
    
    

    
    
    
        
    T_1= TimeWindow[0]
    T_2= TimeWindow[1]
        

        
    for i_bin in range(num_bin):
    
        T_1= TimeWindow[0]+i_bin*BinSize
        T_2= TimeWindow[0]+(i_bin+1)*BinSize

        raw_events=np.array([stim_on+T_1, stim_on+T_2]).T
        events=raw_events
            
            
        raw_events_2=np.array([first_move+T_1, first_move+T_2]).T
        events_2=raw_events_2
            
            
        raw_events_3=np.array([feedback+T_1, feedback+T_2]).T
        events_3=raw_events_3
        #num_trial_cond=len(events[:,0])
        # spike_count.shape(num_neuron,num_trial_cond)
        #  cluster_id.shape(num_neuron,1)
        
        spike_count, cluster_id = get_spike_counts_in_bins(spikes[probe]['times'],spikes[probe]['clusters'],events)
        # firing_rate.shape=(num_neuron,num_bin,num_condition) 
        #count_number[i_bin]=   np.nansum(spike_count,axis=1)
            

        spike_rate[:,:,i_bin]=   spike_count/(T_2-T_1)
        #num_trial_cond=len(events[:,0])
        # spike_count.shape(num_neuron,num_trial_cond)
        #  cluster_id.shape(num_neuron,1)
        
        
        

        
        
        spike_count_2, cluster_id = get_spike_counts_in_bins(spikes[probe]['times'],spikes[probe]['clusters'],events_2)
            # firing_rate.shape=(num_neuron,num_bin,num_condition) 
            #count_number[i_bin]=   np.nansum(spike_count,axis=1)
            
        spike_rate_2[:,:,i_bin]=   spike_count_2/(T_2-T_1)
        #num_trial_cond=len(events[:,0])
        # spike_count.shape(num_neuron,num_trial_cond)
        #  cluster_id.shape(num_neuron,1)
        

        spike_count_3, cluster_id = get_spike_counts_in_bins(spikes[probe]['times'],spikes[probe]['clusters'],events_3)
            # firing_rate.shape=(num_neuron,num_bin,num_condition) 
            #count_number[i_bin]=   np.nansum(spike_count,axis=1)
            
        spike_rate_3[:,:,i_bin]=   spike_count_3/(T_2-T_1)
        #num_trial_cond=len(events[:,0])
        # spike_count.shape(num_neuron,num_trial_cond)
        #  cluster_id.shape(num_neuron,1)
        
        
        
        
        
    
    # rate_1=(num_neuron,)
    #rate_1=np.nanmean(np.nanmean(firing_rate,axis=1),axis=1)
    #rate_1=(np.nanmean(firing_rate[:,:,0],axis=1)
    area_label_1=clusters[probe]['atlas_id'][cluster_id] 

    

     
    
    # only include units pass single unit QC criterion
    ks2_id=np.zeros(0)
    for i in range(len(cluster_id)):
        if clusters[probe]['metrics']['amp_median'][cluster_id[i]]>50/1000000:
            if clusters[probe]['metrics']['slidingRP_viol'][cluster_id[i]]==1 and clusters[probe]['metrics']['noise_cutoff'][cluster_id[i]]<20:
                ks2_id=np.append(ks2_id,[i])
    ks2_id=ks2_id.astype(int)        
        
    #included_units=np.intersect1d(rate_good, ks2_id)
    #included_units=rate_good
    included_units=ks2_id
    included_units=included_units.astype(int)
    
    ######### rate=(num_neuron,trials,time_bin) firing rate aligned to stimulus onset ############
    ######### rate_2=(num_neuron,trials,time_bin) firing rate aligned to first-movement onset ############
    ######### rate_3=(num_neuron,trials,time_bin) firing rate aligned to feedback onset ############
    rate=spike_rate[included_units,:,:]
    rate_2=spike_rate_2[included_units,:,:]
    rate_3=spike_rate_3[included_units,:,:]
    
    area_label=area_label_1[included_units]

    return rate, rate_2, rate_3, block, choice, contrast_L, contrast_R


# In[4]:


####### Example session ###############
eid='15f742e1-1043-45c9-9504-f1e8a53c1744'


# In[9]:


rate_stim_1, rate_move_1, rate_feedback_1, block, choice, contrast_L, contrast_R = single_trial_PSTH(eid, 'probe00')


# In[10]:


rate_stim_2, rate_move_2, rate_feedback_2, block, choice, contrast_L, contrast_R = single_trial_PSTH(eid, 'probe01')


# In[11]:


rate_stim=np.append(rate_stim_1,rate_stim_2,axis=0)
rate_move=np.append(rate_move_1,rate_move_2,axis=0)
rate_feedback=np.append(rate_feedback_1,rate_feedback_2,axis=0)


# In[12]:


########### test p-value of single-cell correlates with visual stim side  ###############
###### Timewindow=[50,100]ms after stimulus onset %%%%%%%%
spike_count=np.nanmean(rate_stim[:,:,45:49],axis=2)
p=get_stim_time_shuffle(spike_count,contrast_L,contrast_R,block,choice,2000)


# In[13]:


import matplotlib.pyplot as plt


# In[50]:


######## Example of single-cell PSTH and associated p-value #############
time_bin = np.arange(-0.4, 0.3, 0.01)
###### neuron 1: (significant)
index_1=48 

PSTH_L_1=np.nanmean(rate_stim[index_1,np.argwhere(contrast_L>0)[:,0],:],axis=0)
PSTH_R_1=np.nanmean(rate_stim[index_1,np.argwhere(contrast_R>0)[:,0],:],axis=0)

###### neuron 2:  (insignificant)
index_2=59

PSTH_L_2=np.nanmean(rate_stim[index_2,np.argwhere(contrast_L>0)[:,0],:],axis=0)
PSTH_R_2=np.nanmean(rate_stim[index_2,np.argwhere(contrast_R>0)[:,0],:],axis=0)



fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True,
                                    figsize=(12, 6))

ax0.set_title('p='+np.array2string(p[index_1]))
ax0.plot(time_bin, PSTH_L_1)
ax0.plot(time_bin, PSTH_R_1)

ax1.set_title('p='+np.array2string(p[index_2]))
ax1.plot(time_bin, PSTH_L_2)
ax1.plot(time_bin, PSTH_R_2)


fig.suptitle('Example of singel-cell PSTH and associated p-value')
plt.show()


# In[ ]:




