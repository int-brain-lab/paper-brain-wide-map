#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 08:58:41 2022

@author: bensonb
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, balanced_accuracy_score


def gini(x, weights=None):
    '''
    Index that measures sparsity across a set of values, x.
    If values in x are roughly constant, then this index is near 0.
    If values in x have a large spread, then this index is near 1.

    Implementation copied from 
    https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python

    Parameters
    ----------
    x : 1-d array of values
        .
    weights : 1-d array, floats
        weightings corresponding to values of x. The default is None.

    Returns
    -------
    float

    '''
    if weights is None:
        weights = np.ones_like(x)
    # Calculate mean absolute deviation in two steps, for weights.
    count = np.multiply.outer(weights, weights)
    mad = np.abs(np.subtract.outer(x, x) * count).sum() / count.sum()
    rmad = mad / np.average(x, weights=weights)
    # Gini equals half the relative mean absolute deviation.
    return 0.5 * rmad


def fix_pd_regions(res):
    '''
    Takes a pandas dataframe and fixes the 'region' column.
    Regions are often lists with single values which makes dataframe
    manipulation difficult.  This function fixes that by removing 
    the list and using the single value

    If regions which are not of type list are not changed

    A region list with multiple elements causes an assertion error

    Parameters
    ----------
    res : pandas dataframe
          must have a column called 'region'
    Returns
    -------
    pandas dataframe

    '''

    print('fixing pd regions')
    res = res.reset_index()
    l = len(res.loc[:,'region'])
    for i, r in enumerate(res.loc[:,'region']):
        if i%2000==0:
            print(f'fixed {i/l:.4f} of table')
        if type(r) is list:
            assert len(r) == 1
            res.loc[i,'region'] = r[0]
    return res


def get_val_from_realsession(reseidreg, value_name, RUN_ID=1):
    '''
    helper function meant to take a pandas dataframe, reseidreg, of decoding results from
    a single eid and a single region.  The value corresponding to the value_name column,
    RUN_ID number, and pseudo_id=-1 is returned.  There should only be one such value, so 
    if the dataframe contains mutliple, then None is returned.  If the value is None, then
    None is returned.       

    Parameters
    ----------
    reseidreg : pandas dataframe
        Decoding results from a single eid and single region. 
        .
    value_name : str
        the name of a column in reseidreg
        .
    RUN_ID=1 : int
        The run id of the desired decoding value.  Decoding is often repeated multiple 
        times to reduce variability, and run id indexes these repetitions. Default is 1.

    Returns
    -------
    numpy array with a single value

    '''

    my_vals = list(reseidreg.loc[(reseidreg['pseudo_id']==-1)&(reseidreg['run_id']==RUN_ID), value_name])
    if (len(my_vals) != 1) or (my_vals[0] is None):
        return None
    return np.array(my_vals[0])


def check_scores(my_preds, my_targets, score_name, real_scores):
    '''
    checks whether the predictions produce the same performance 
    scores as those that are given in the real_scores array.  
    Compares predictions (my_preds) and targets (my_targets) to produce the
    desired score_name and returns True if calculated scores all match the
    recored scores

    Parameters
    ----------
    my_preds : list of 2-d arrays
        2-d arrays are decoder predictions across trials.  Assumed that 
        second dimension has size 1 i.e. not wheel decoding.  The list indexes
        across run ids
    my_targets : 2-d array
        Same format as my_preds, but not a list (targets are the same for all run ids).
    score_name : str, 'balanced_acc_test' or 'R2_test'
        These are the two test statistics used to quantify BWM decoding performance: 
        balanced_accuracy_score and r2_score from sklearn.metrics. 
    real_scores : 1-d array
        The recored scores of decoding performance across run id.

    Returns
    -------
    boolean

    '''

    my_targets_flat = my_targets[:,0]
    my_preds_flat = [my_preds[pi][:,0] for pi in range(my_preds.shape[0])]
    assert len(my_targets_flat.shape)==1
    assert np.all(np.array([len(p.shape) for p in my_preds_flat])==1)
    
    if score_name == 'balanced_acc_test':
        calc_score = lambda x: balanced_accuracy_score(my_targets_flat, x)
    elif score_name == 'R2_test':
        calc_score = lambda x: r2_score(my_targets_flat, x)
    my_calc_real_scores = [calc_score(p) for p in my_preds_flat]
    isequal_scores = [np.isclose(my_calc_real_scores[i],real_scores[i]) for i in range(len(real_scores))]
    return np.all(np.array(isequal_scores))


def create_pdtable_from_raw(res, 
                            score_name='balanced_acc_test',
                            N_PSEUDO=200, N_RUN=10, 
                            N_PSEUDO_LOWER_THRESH = np.infty,
                            RETURN_X_Y=False,
                            SCALAR_PER_TRIAL=True,
                            SAVE_REGRESSORS=True):
    '''
    Takes formatted outputs of decoders and aggregates important values for post-processing
    including subject, eid, region, test statistic, p-value, median of null distribution,
    number of units used for decoding, fraction of weights which exceed a threshold of 0.1, 
    and the gini index of the weights to capture sparsity.
    Optionally returns a secondary table which also includes all the regressors, targets,
    predictions, and weights.

    Parameters
    ----------
    res : pandas DataFrame 
        formatted as done in the output of 04_format_slurm.py.
    score_name : str, optional
        'balanced_acc_test' or 'R2_test'. The default is 'balanced_acc_test'.
    N_PSEUDO : int, optional
        from settings.py. The default is 200.
    N_RUN : int, optional
        from settings.py. The default is 10.
    N_PSEUDO_LOWER_THRESH : scalar, optional
        if finite, allows processing of outputs when not all N_PSEUDO nulls are present. 
        The default is np.infty.
    RETURN_X_Y : bool
        Only works for scalar values per trial, e.g. not wheel-speed. 
        returns an additional table with regressors, targets, and predictions
        The default is False.
    SCALAR_PER_TRIAL : bool
        set to False if decoding multiple values per trial e.g. wheel decoding
        The default is True.
    SAVE_REGRESSORS : bool
        if RETURN_X_Y then the returned xy-table's regressors columns will be empty if
        SAVE_REGRESSORS if False.  Either way, the same checks are done on regressors',
        representation, they are just not saved.
        The default is True.

    Returns
    -------
    pandas DataFrame 
        summary of all decoding test statistics, p-values, weight analysis, and units.
    pandas DataFrame
        Optional, only returned if RETURN_X_Y.
        summary of add decoder regressors, targets, and weights

    '''
    if not score_name in ['balanced_acc_test', 'R2_test']:
        raise NotImplementedError('this score is not implemented')
    
    res_table = []
    xy_table = []
    
    for eid in np.unique(res['eid']):
        print(f'working on {eid}')
        reseid = res.loc[res['eid']==eid]
        subject = np.unique(reseid['subject'])
        assert len(subject) == 1
        subject = subject[0]
        
        #print(reseid['region'])
        for reg in np.unique(reseid['region']):
            
            reseidreg = reseid.loc[reseid['region']==reg]
            eidreg_probes = np.unique(reseidreg['probe'])
            assert len(eidreg_probes) == 1                
            
            pids = np.sort(np.unique(reseidreg['pseudo_id']))
            #print(reseidreg.head())
            if len(pids) == N_PSEUDO+1:
                assert pids[0] == -1
                assert np.all(pids[1:] == np.arange(1,N_PSEUDO+1))
                real_scores = [get_val_from_realsession(reseidreg, 
                                                        score_name, 
                                                        RUN_ID=runid) for runid in range(1,N_RUN+1)]
                #real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
                #assert len(real_scores) == N_RUN
            # elif len(pids) >= N_PSEUDO_LOWER_THRESH+1 and pids[0] == -1:
            #     print('not full pseudo_ids', len(pids))
            #     real_scores = reseidreg.loc[reseidreg['pseudo_id']==-1,score_name]
            #     assert len(real_scores) >= N_RUN - 1
                
            else:
                print(f'skipping eid ({eid}) and region ({reg}) because only {len(pids)} pseudo_ids are present')
                continue
            
            ws = list(reseidreg.loc[reseidreg['pseudo_id']==-1, 'weights'])
            my_weights = np.stack(ws)
            assert len(ws)==N_RUN
            ws = np.abs(np.ndarray.flatten(my_weights))
            #print(ws)
            frac_lg_w = np.mean(ws > 0.1)#1.0/len(ws))
            gini_w = gini(ws)
            # 10 repeats of decoding to reduce variance
            score = np.mean(real_scores)
            
            # include real score in null scores
            n_runs_per_p = [len(np.array(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name])) for pid in pids]#[1:]
            assert np.all(np.array(n_runs_per_p)==N_RUN)
            p_scores = [np.mean(reseidreg.loc[reseidreg['pseudo_id']==pid,score_name]) for pid in pids]#[1:]
            if np.any(np.isnan(p_scores)):
                print(f'skipping eid ({eid}) and region ({reg}) because {np.sum(np.isnan(p_scores))} scores are nan')
                continue
            
            median_null = np.median(p_scores)
            pval = np.mean(np.array(p_scores)>=score)
            n_units = np.array(reseidreg.loc[reseidreg['pseudo_id']==-1,'N_units'])
            assert np.all(n_units == n_units[0])
            n_units = n_units[0]
            
            if RETURN_X_Y:
                
                # load regressors
                my_regressors = get_val_from_realsession(reseidreg, 'regressors')
                if my_regressors is None:
                    print(f'skipping eid ({eid}) and region ({reg}) because regressors are not present')
                    continue

                # load cluster uuids
                my_cuuids = [get_val_from_realsession(reseidreg,
                                                     'cluster_uuids',
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)]
                if np.any(np.array([c is None for c in my_cuuids])):
                    print(f'skipping eid ({eid}) and region ({reg}) becuase cluster uuids are not present')
                    continue
                assert np.all([np.all(np.array(my_cuuids[0])==np.array(c)) for c in my_cuuids])
                my_cuuids = my_cuuids[0]
              
                # load targets
                my_targets = get_val_from_realsession(reseidreg, 'target')
                if my_targets is None:
                    print(f'skipping eid ({eid}) and region ({reg}) because targets are not present')
                    continue
                
                # load predictions
                my_preds = [get_val_from_realsession(reseidreg, 
                                                     'prediction', 
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)] 
                my_preds = np.stack(my_preds)
                assert my_preds.shape[0] == N_RUN
                
                if np.any(np.array([mps is None for mps in my_preds])):
                    print(f'skipping eid ({eid}) and region ({reg}) because predictions are not present')
                    continue

                my_intercepts = np.stack(list(reseidreg.loc[reseidreg['pseudo_id']==-1, 'intercepts']))
                my_idxes = np.stack(list(reseidreg.loc[reseidreg['pseudo_id']==-1, 'idxes_test']))

                # load parameters
                my_params = [get_val_from_realsession(reseidreg, 
                                                     'params', 
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)] 
                if np.any(np.array([mps is None for mps in my_params])):
                    print(f'skipping eid ({eid}) and region ({reg}) because params are not present')
                    continue
                my_params = [[[(k,mp_fold[k]) for k in mp_fold.keys()] for mp_fold in mp_run] for mp_run in my_params]
                my_params = np.stack(my_params)
                assert my_params.shape[0] == N_RUN
                
                # load mask
                my_masks = [get_val_from_realsession(reseidreg, 
                                                     'mask', 
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)]
                my_masks_trials_and_targets = [get_val_from_realsession(reseidreg,
                                                     'mask_trials_and_targets',
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)]
                my_masks_diagnostics = [get_val_from_realsession(reseidreg,
                                                     'mask_diagnostics',
                                                     RUN_ID=runid) for runid in range(1,N_RUN+1)] 
                
                if np.any(np.array([m is None for m in my_masks])):
                    print(f'skipping eid ({eid}) and region ({reg}) because mask is not present')
                    continue
                assert np.all(np.array(my_masks)==my_masks[0])
                #print(type(my_masks[0]), my_masks[0].shape)
                my_mask = str(my_masks[0])
                my_mask = [int(my_mask[mi]) for mi in range(len(my_mask))]
                assert np.all(np.unique(my_mask)==np.array([0,1]))
                
                
                # check arrays
                assert my_targets.shape == my_preds[0].shape
                assert my_regressors.shape[0] == my_targets.shape[0]
                assert np.sum(my_mask) == my_targets.shape[0]
                if SAVE_REGRESSORS: # TODO, hack for wheel, check why this fails for wheel
                    assert len(my_cuuids) == my_regressors.shape[-1]
                #if np.any(np.array([len(np.unique(my_preds[pi]))==1 for pi in range(my_preds.shape[0])])):
                    #print(f'at least one pred is constant {eid} {reg}', )
                    #continue
                
                if SCALAR_PER_TRIAL:
                    check_preds = (my_preds > 0.5) if score_name == 'balanced_acc_test' else my_preds
                    
                    #print('############# debugging ###################')
                    #my_preds_flat = [my_preds[pi][:,0]>0.5 for pi in range(my_preds.shape[0])]
                    #my_targets_flat = my_targets[:,0]
                    #assert len(my_targets_flat.shape)==1
                    #assert np.all(np.array([len(p.shape) for p in my_preds_flat])==1)

                    #if score_name == 'balanced_acc_test':
                    #calc_score = lambda x: balanced_accuracy_score(my_targets_flat, x)
                    #elif score_name == 'R2_test':
                    #    calc_score = lambda x: r2_score(my_targets_flat, x)
                    
                    #print('shapes')
                    #print(my_preds_flat[0].shape, my_targets_flat.shape)
                    #print('errors')
                    #errs = (my_preds_flat[0] != my_targets_flat)
                    #print(np.nonzero(errs))
                    #print(my_preds_flat[0])
                    #print(real_scores)
                    
                    #my_calc_real_scores = [calc_score(p) for p in my_preds_flat]
                    #print(my_calc_real_scores)
                    #isequal_scores = [np.isclose(my_calc_real_scores[i],real_scores[i]) for i in range(len(real_scores))]
                    #print('return', np.all(np.array(isequal_scores)))
                    
                    
                    
                    if not check_scores(check_preds, 
                                        my_targets, 
                                        score_name, 
                                        real_scores):
                        #print(my_preds, check_preds, my_targets, real_scores)
                        raise ValueError('recorded scores do not match calculated scores')
                
            res_table.append([subject,
                              eid,
                              reg,
                              score,
                              pval,
                              median_null,
                              n_units,
                              frac_lg_w,
                              gini_w])
            if RETURN_X_Y:
                if not SAVE_REGRESSORS:
                    my_regressors = []
                xy_table.append([f'{eid}_{reg}', 
                                 my_regressors, 
                                 my_targets, 
                                 my_preds,
                                 my_mask,
                                 (my_masks_trials_and_targets, my_masks_diagnostics),
                                 my_weights,
                                 my_intercepts,
                                 my_idxes,
                                 my_params,
                                 my_cuuids])
                
    res_table = pd.DataFrame(res_table, columns=['subject',
                                                 'eid',
                                                 'region',
                                                 'score',
                                                 'p-value',
                                                 'median-null',
                                                 'n_units',
                                                 'frac_large_w',
                                                 'gini_w'])
    
    if RETURN_X_Y:
        xy_table = pd.DataFrame(xy_table, columns=['eid_region',
                                                   'regressors',
                                                   'targets',
                                                   'predictions',
                                                   'mask',
                                                   'mask_diagnostics',
                                                   'weights',
                                                   'intercepts',
                                                   'idxes',
                                                   'params',
                                                   'cluster_uuids'])
        return res_table, xy_table
    
    return res_table
