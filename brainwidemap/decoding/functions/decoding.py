import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import linear_model as sklm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from tqdm import tqdm
from behavior_models.models.utils import format_data as format_data_mut
from behavior_models.models.utils import format_input as format_input_mut

from ibllib.atlas import BrainRegions

from brainwidemap.decoding.functions.balancedweightings import balanced_weighting
from brainwidemap.decoding.functions.balancedweightings import get_balanced_weighting
from brainwidemap.decoding.functions.process_inputs import build_predictor_matrix
from brainwidemap.decoding.functions.process_inputs import select_ephys_regions
from brainwidemap.decoding.functions.process_inputs import preprocess_ephys
from brainwidemap.decoding.functions.process_targets import compute_beh_target
from brainwidemap.decoding.functions.process_targets import compute_target_mask
from brainwidemap.decoding.functions.process_targets import get_target_data_per_trial_wrapper
from brainwidemap.decoding.functions.utils import save_region_results
from brainwidemap.decoding.functions.utils import get_save_path
from brainwidemap.decoding.functions.nulldistributions import generate_null_distribution_session
from brainwidemap.decoding.functions.process_targets import check_bhv_fit_exists
from brainwidemap.decoding.functions.process_targets import optimal_Bayesian


def fit_eid(neural_dict, trials_df, trials_mask, metadata, dlc_dict=None, pseudo_ids=[-1], **kwargs):
    """High-level function to decode a given target variable from brain regions for a single eid.

    Parameters
    ----------
    neural_dict : dict
        keys: 'spk_times', 'spk_clu', 'clu_regions', 'clu_qc', 'clu_df'
    trials_df : pd.DataFrame
        columns: 'choice', 'feedback', 'pLeft', 'firstMovement_times', 'stimOn_times',
        'feedback_times'
    trials_mask: pd.Series
        Boolean mask indicating good trials
    metadata : dict
        'eid', 'eid_train', 'subject', 'probes'
    dlc_dict: dict, optional
        keys: 'times', 'values'
    pseudo_ids : array-like
        whether to compute a pseudosession or not. if pseudo_id=-1, the true session is considered.
        if pseudo_id>0, a pseudo session is used. cannot be 0.
    kwargs
        date : str
            date string for specifying filenames
        target : str
            single-bin targets: 'pLeft' | 'signcont' | 'choice' | 'feedback'
            multi-bin targets: 'wheel-vel' | 'wheel-speed' | 'pupil' | '[l/r]-paw-pos'
                | '[l/r]-paw-vel' | '[l/r]-paw-speed' | '[l/r]-whisker-me'
        align_time : str
            event in trial on which to align intervals
            'firstMovement_times' | 'stimOn_times' | 'feedback_times'
        time_window : tuple
            (window_start, window_end), relative to align_time
        binsize : float
            size of bins in seconds for multi-bin decoding
        n_bins_lag : int
            number of lagged bins to use for predictors for multi-bin decoding
        estimator : sklearn.linear_model object
            sklm.Lasso | sklm.Ridge | sklm.LinearRegression | sklm.LogisticRegression
        hyperparam_grid : dict
            regularization values to search over
        n_runs : int
            number of independent runs performed. this was added after variability was observed
            across runs.
        shuffle : bool
            True for interleaved cross-validation, False for contiguous blocks
        min_units : int
            minimum units per region to use for decoding
        min_behav_trials : int
            minimum number of trials (after filtering) that must be present to proceed with fits
        min_rt : float
            minimum reaction time per trial; can be used to filter out trials with negative
            reaction times
        no_unbias : bool
            True to remove unbiased trials; False to keep
        output_path : str
            absolute path where decoding fits are saved
        add_to_saving_path : str
            additional string to append to filenames

    """

    print(f'Working on eid: %s' % metadata['eid'])
    filenames = []  # this will contain paths to saved decoding results for this eid

    if kwargs['use_imposter_session'] and not kwargs['stitching_for_imposter_session']:
        trials_df = trials_df[:int(kwargs['max_number_trials_when_no_stitching_for_imposter_session'])]

    if 0 in pseudo_ids:
        raise ValueError(
            'pseudo id can be -1 (actual session) or strictly greater than 0 (pseudo session)')

    if not np.all(np.sort(pseudo_ids) == pseudo_ids):
        raise ValueError('pseudo_ids must be sorted')

    if kwargs['model'] == optimal_Bayesian and np.any(trials_df.probabilityLeft.values[:90] != 0.5):
        raise ValueError(
            f'The optimal Bayesian model assumes 90 unbiased trials at the beginning of the '
            f'session, which is not the case here.')

    # check if is trained
    eids_train = (
        [metadata['eid']] if 'eids_train' not in metadata.keys() else metadata['eids_train'])
    if 'eids_train' not in metadata.keys():
        metadata['eids_train'] = eids_train
    elif metadata['eids_train'] != eids_train:
        raise ValueError(f'eids_train are not supported yet. If you do not understand this error, '
                         f'just take out the eids_train key in the metadata to solve it')

    if isinstance(kwargs['model'], str):
        import pickle
        from braindelphi.params import INTER_INDIVIDUAL_PATH
        inter_individual = pickle.load(open(INTER_INDIVIDUAL_PATH.joinpath(kwargs['model']), 'rb'))
        if metadata['eid'] not in inter_individual.keys():
            logging.exception('no inter individual model found')
            return filenames
        inter_indiv_model_specifications = inter_individual[metadata['eid']]
        print('winning interindividual model is %s' % inter_indiv_model_specifications['model_name'])
        if inter_indiv_model_specifications['model_name'] not in kwargs['modeldispatcher'].values():
            logging.exception('winning inter individual model is LeftKernel or RightKernel')
            return filenames
        kwargs['model'] = {v: k for k, v in kwargs['modeldispatcher'].items()}[inter_indiv_model_specifications['model_name']]
        kwargs['model_parameters'] = inter_indiv_model_specifications['model_parameters']
    else:
        kwargs['model_parameters'] = None
        # train model if not trained already
        if kwargs['model'] != optimal_Bayesian and kwargs['model'] is not None:
            side, stim, act, _ = format_data_mut(trials_df)
            stimuli, actions, stim_side = format_input_mut([stim], [act], [side])
            behmodel = kwargs['model'](
                kwargs['behfit_path'], np.array(metadata['eids_train']), metadata['subject'],
                actions, stimuli, stim_side, single_zeta=True)
            istrained, _ = check_bhv_fit_exists(
                metadata['subject'], kwargs['model'], metadata['eids_train'],
                kwargs['behfit_path'], modeldispatcher=kwargs['modeldispatcher'], single_zeta=True)
            if not istrained:
                behmodel.load_or_train(remove_old=False)

    target_distribution = get_balanced_weighting(trials_df, metadata, **kwargs)

    # get target values
    if kwargs['target'] in ['pLeft', 'signcont', 'strengthcont', 'choice', 'feedback']:
        target_vals_list, target_vals_to_mask = compute_beh_target(
            trials_df, metadata, return_raw=True, **kwargs)
        target_mask = compute_target_mask(
            target_vals_to_mask, kwargs['exclude_trials_within_values'])

    else:
        if dlc_dict is None or dlc_dict['times'] is None or dlc_dict['values'] is None:
            raise ValueError('dlc_dict does not contain any data')
        _, target_vals_list, target_mask = get_target_data_per_trial_wrapper(
            target_times=dlc_dict['times'],
            target_vals=dlc_dict['values'],
            trials_df=trials_df,
            align_event=kwargs['align_time'],
            align_interval=kwargs['time_window'],
            binsize=kwargs['binsize'])

    mask = trials_mask & target_mask

    if sum(mask) <= kwargs['min_behav_trials']:
        msg = 'session contains %i trials, below the threshold of %i' % (
            sum(mask), kwargs['min_behav_trials'])
        logging.exception(msg)
        return filenames

    # select brain regions from beryl atlas to loop over
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict['clu_regions'], mapping='Beryl')
    regions = (
        [[k] for k in np.unique(beryl_reg)] if kwargs['single_region'] else [np.unique(beryl_reg)])

    for region in tqdm(regions, desc='Region: ', leave=False):

        # pull spikes from this region out of the neural data
        reg_clu_ids = select_ephys_regions(neural_dict, beryl_reg, region, **kwargs)

        # skip region if there are not enough units
        n_units = len(reg_clu_ids)
        if n_units < kwargs['min_units']:
            continue

        # bin spikes from this region for each trial
        msub_binned, cl_inds_used = preprocess_ephys(reg_clu_ids, neural_dict, trials_df, **kwargs)
        cl_uuids_used = list(neural_dict['clu_df'].iloc[cl_inds_used]['uuids'])

        # make design matrix
        bins_per_trial = msub_binned[0].shape[0]
        Xs = (
            msub_binned if bins_per_trial == 1
            else [build_predictor_matrix(s, kwargs['n_bins_lag']) for s in msub_binned]
        )

        for pseudo_id in pseudo_ids:
            fit_results = []

            # save out decoding results
            save_path = get_save_path(
                pseudo_id, metadata['subject'], metadata['eid'], 'ephys',
                probe=metadata['probe_name'],
                region=str(np.squeeze(region)) if kwargs['single_region'] else 'allRegions',
                output_path=kwargs['neuralfit_path'],
                time_window=kwargs['time_window'],
                date=kwargs['date'],
                target=kwargs['target'],
                add_to_saving_path=kwargs['add_to_saving_path']
            )
            if os.path.exists(save_path):
                print(
                    f'results for region {region}, pseudo_id {pseudo_id} already exist at '
                    f'{save_path}')
                filenames.append(save_path)
                continue

            # create pseudo/imposter session when necessary and corresponding mask
            if pseudo_id > 0:
                if bins_per_trial == 1:
                    controlsess_df = generate_null_distribution_session(
                        trials_df, metadata, **kwargs)
                    controltarget_vals_list, controltarget_vals_to_mask = compute_beh_target(
                        controlsess_df, metadata, return_raw=True, **kwargs)
                    controltarget_mask = compute_target_mask(
                        controltarget_vals_to_mask, kwargs['exclude_trials_within_values'])
                    control_mask = trials_mask & controltarget_mask
                else:
                    imposter_df = kwargs['imposter_df'].copy()
                    # remove current eid from imposter sessions
                    df_clean = imposter_df[imposter_df.eid != metadata['eid']].reset_index()
                    # randomly select imposter trial to start sequence
                    n_trials = trials_df.index.size
                    total_imposter_trials = df_clean.shape[0]
                    idx_beg = np.random.choice(total_imposter_trials - n_trials)
                    controlsess_df = df_clean.iloc[idx_beg:idx_beg + n_trials]
                    # grab target values from this dataframe
                    controltarget_vals_list = list(controlsess_df[kwargs['target']].to_numpy())
                    control_mask = mask

                save_predictions = kwargs.get('save_predictions_pseudo', kwargs['save_predictions'])
            else:
                control_mask = mask
                save_predictions = kwargs['save_predictions']

            # run decoders
            for i_run in range(kwargs['n_runs']):

                if kwargs['quasi_random']:
                    if pseudo_id == -1:
                        rng_seed = i_run
                    else:
                        rng_seed = pseudo_id * kwargs['n_runs'] + i_run
                else:
                    rng_seed = None

                if pseudo_id == -1:
                    # original session
                    ys_wmask = [target_vals_list[m] for m in np.squeeze(np.where(mask))]
                    Xs_wmask = [Xs[m] for m in np.squeeze(np.where(mask))]
                else:
                    # session for null dist
                    ys_wmask = [controltarget_vals_list[m] for m in np.squeeze(np.where(control_mask))]
                    Xs_wmask = [Xs[m] for m in np.squeeze(np.where(control_mask))]                

                #if i_run == 0:
                #    print('target, mask, and target after applying mask:', target_vals_list)
                #    print(mask)
                #    print(trials_mask)
                #    print(target_mask)
                #    print(ys_wmask)

                fit_result = decode_cv(
                    ys=ys_wmask,
                    Xs=Xs_wmask,                    
                    estimator=kwargs['estimator'],
                    use_openturns=kwargs['use_openturns'],
                    target_distribution=target_distribution,
                    bin_size_kde=kwargs['bin_size_kde'],
                    balanced_continuous_target=kwargs['balanced_continuous_target'],
                    estimator_kwargs=kwargs['estimator_kwargs'],
                    hyperparam_grid=kwargs['hyperparam_grid'],
                    save_binned=kwargs['save_binned'] if pseudo_id==-1 else False,
                    save_predictions=save_predictions,
                    shuffle=kwargs['shuffle'],
                    balanced_weight=kwargs['balanced_weight'],
                    rng_seed=rng_seed,
                )
                fit_result['mask'] = mask
                fit_result['mask_trials_and_targets'] = [trials_mask, target_mask]
                fit_result['mask_diagnostics'] = kwargs['trials_mask_diagnostics']
                fit_result['df'] = trials_df if pseudo_id == -1 else controlsess_df
                fit_result['pseudo_id'] = pseudo_id
                fit_result['run_id'] = i_run
                fit_result['cluster_uuids'] = cl_uuids_used
                fit_results.append(fit_result)

            filename = save_region_results(
                fit_result=fit_results,
                pseudo_id=pseudo_id,
                subject=metadata['subject'],
                eid=metadata['eid'],
                probe=metadata['probe_name'],
                region=region,
                n_units=n_units,
                save_path=save_path
            )

            filenames.append(filename)

    print(f'Finished eid: %s' % metadata['eid'])

    return filenames


def decode_cv(
        ys,
        Xs,
        estimator,
        estimator_kwargs,
        use_openturns,
        target_distribution,
        bin_size_kde,
        balanced_continuous_target=True,
        balanced_weight=False,
        hyperparam_grid=None,
        test_prop=0.2,
        n_folds=5,
        save_binned=False,
        save_predictions=True,
        verbose=False,
        shuffle=True,
        outer_cv=True,
        rng_seed=None,
        normalize_input=False,
        normalize_output=False,
):
    """Regresses binned neural activity against a target, using a provided sklearn estimator.

    Parameters
    ----------
    ys : list of arrays or np.ndarray or pandas.Series
        targets; if list, each entry is an array of targets for one trial. if 1D numpy array, each
        entry is treated as a single scalar for one trial. if pd.Series, trial number is the index
        and teh value is the target.
    Xs : list of arrays or np.ndarray
        predictors; if list, each entry is an array of neural activity for one trial. if 2D numpy
        array, each row is treated as a single vector of ativity for one trial, i.e. the array is
        of shape (n_trials, n_neurons)
    estimator : sklearn.linear_model object
        estimator from sklearn which provides .fit, .score, and .predict methods. CV estimators
        are NOT SUPPORTED. Must be a normal estimator, which is internally wrapped with
        GridSearchCV
    estimator_kwargs : dict
        additional arguments for sklearn estimator
    use_openturns : bool
    target_distribution : ?
        ?
    bin_size_kde : float
        ?
    balanced_weight : ?
        ?
    balanced_continuous_target : ?
        ?
    hyperparam_grid : dict
        key indicates hyperparameter to grid search over, and value is an array of nodes on the
        grid. See sklearn.model_selection.GridSearchCV : param_grid for more specs.
        Defaults to None, which means no hyperparameter estimation or GridSearchCV use.
    test_prop : float
        proportion of data to hold out as the test set after running hyperparameter tuning; only
        used if `outer_cv=False`
    n_folds : int
        Number of folds for cross-validation during hyperparameter tuning; only used if
        `outer_cv=True`
    save_binned : bool
        True to put the regressors Xs into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    save_predictions : bool
        True to put the model predictions into the output dictionary.
        Can cause file bloat if saving outputs.
        Note: this function does not actually save any files!
    shuffle : bool
        True for interleaved cross-validation, False for contiguous blocks
    outer_cv: bool
        Perform outer cross validation such that the testing spans the entire dataset
    rng_seed : int
        control data splits
    verbose : bool
        Whether you want to hear about the function's life, how things are going, and what the
        neighbor down the street said to it the other day.
    normalize_output : bool
        True to take out the mean across trials of the output
    normalize_input : bool
        True to take out the mean across trials of the input; average is taken across trials for
        each unit (one average per unit is computed)

    Returns
    -------
    dict
        Dictionary of fitting outputs including:
            - Regression score (from estimator)
            - Decoding coefficients
            - Decoding intercept
            - Per-trial target values (copy of tvec)
            - Per-trial predictions from model
            - Input regressors (optional, see Xs argument)

    """

    # transform target data into standard format: list of np.ndarrays
    if isinstance(ys, np.ndarray):
        # input is single numpy array
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, list) and ys[0].shape == ():
        # input is list of float instead of list of np.ndarrays
        ys = [np.array([y]) for y in ys]
    elif isinstance(ys, pd.Series):
        # input is a pandas Series
        ys = ys.to_numpy()
        ys = [np.array([y]) for y in ys]

    # transform neural data into standard format: list of np.ndarrays
    if isinstance(Xs, np.ndarray):
        Xs = [x[None, :] for x in Xs]

    # initialize containers to save outputs
    n_trials = len(Xs)
    bins_per_trial = len(Xs[0])
    scores_test, scores_train = [], []
    idxes_test, idxes_train = [], []
    weights, intercepts, best_params = [], [], []
    predictions = [None for _ in range(n_trials)]
    predictions_to_save = [None for _ in range(n_trials)]  # different for logistic regression

    # split the dataset in two parts, train and test
    # when shuffle=False, the method will take the end of the dataset to create the test set
    if rng_seed is not None:
        np.random.seed(rng_seed)
    indices = np.arange(n_trials)
    if outer_cv:
        outer_kfold = KFold(n_splits=n_folds, shuffle=shuffle).split(indices)
    else:
        outer_kfold = iter([train_test_split(indices, test_size=test_prop, shuffle=shuffle)])

    # scoring function; use R2 for linear regression, accuracy for logistic regression
    scoring_f = balanced_accuracy_score if (estimator == sklm.LogisticRegression) else r2_score

    # Select either the GridSearchCV estimator for a normal estimator, or use the native estimator
    # in the case of CV-type estimators
    if estimator == sklm.RidgeCV \
            or estimator == sklm.LassoCV \
            or estimator == sklm.LogisticRegressionCV:
        raise NotImplementedError('the code does not support a CV-type estimator for the moment.')
    else:
        # loop over outer folds
        for train_idxs_outer, test_idxs_outer in outer_kfold:
            # outer fold data split
            # X_train = np.vstack([Xs[i] for i in train_idxs])
            # y_train = np.concatenate([ys[i] for i in train_idxs], axis=0)
            # X_test = np.vstack([Xs[i] for i in test_idxs])
            # y_test = np.concatenate([ys[i] for i in test_idxs], axis=0)
            X_train = [Xs[i] for i in train_idxs_outer]
            y_train = [ys[i] for i in train_idxs_outer]
            X_test = [Xs[i] for i in test_idxs_outer]
            y_test = [ys[i] for i in test_idxs_outer]

            # now loop over inner folds
            idx_inner = np.arange(len(X_train))
            inner_kfold = KFold(n_splits=n_folds, shuffle=shuffle).split(idx_inner)

            key = list(hyperparam_grid.keys())[0]  # TODO: make this more robust
            r2s = np.zeros([n_folds, len(hyperparam_grid[key])])
            for ifold, (train_idxs_inner, test_idxs_inner) in enumerate(inner_kfold):

                # inner fold data split
                X_train_inner = np.vstack([X_train[i] for i in train_idxs_inner])
                y_train_inner = np.concatenate([y_train[i] for i in train_idxs_inner], axis=0)
                X_test_inner = np.vstack([X_train[i] for i in test_idxs_inner])
                y_test_inner = np.concatenate([y_train[i] for i in test_idxs_inner], axis=0)

                # normalize inputs/outputs if requested
                mean_X_train_inner = X_train_inner.mean(axis=0) if normalize_input else 0
                X_train_inner = X_train_inner - mean_X_train_inner
                X_test_inner = X_test_inner - mean_X_train_inner
                mean_y_train_inner = y_train_inner.mean(axis=0) if normalize_output else 0
                y_train_inner = y_train_inner - mean_y_train_inner

                for i_alpha, alpha in enumerate(hyperparam_grid[key]):

                    # compute weight for each training sample if requested
                    # (esp necessary for classification problems with imbalanced classes)
                    if balanced_weight:
                        sample_weight = balanced_weighting(
                            vec=y_train_inner,
                            continuous=balanced_continuous_target,
                            use_openturns=use_openturns,
                            bin_size_kde=bin_size_kde,
                            target_distribution=target_distribution)
                    else:
                        sample_weight = None

                    # initialize model
                    model_inner = estimator(**{**estimator_kwargs, key: alpha})
                    # fit model
                    model_inner.fit(X_train_inner, y_train_inner, sample_weight=sample_weight)
                    # evaluate model
                    pred_test_inner = model_inner.predict(X_test_inner) + mean_y_train_inner
                    r2s[ifold, i_alpha] = scoring_f(y_test_inner, pred_test_inner)

            # select model with best hyperparameter value evaluated on inner-fold test data;
            # refit/evaluate on all inner-fold data
            r2s_avg = r2s.mean(axis=0)

            # normalize inputs/outputs if requested
            X_train_array = np.vstack(X_train)
            mean_X_train = X_train_array.mean(axis=0) if normalize_input else 0
            X_train_array = X_train_array - mean_X_train

            y_train_array = np.concatenate(y_train, axis=0)
            mean_y_train = y_train_array.mean(axis=0) if normalize_output else 0
            y_train_array = y_train_array - mean_y_train

            # compute weight for each training sample if requested
            if balanced_weight:
                sample_weight = balanced_weighting(
                    vec=y_train_array,
                    continuous=balanced_continuous_target,
                    use_openturns=use_openturns,
                    bin_size_kde=bin_size_kde,
                    target_distribution=target_distribution)
            else:
                sample_weight = None

            # initialize model
            best_alpha = hyperparam_grid[key][np.argmax(r2s_avg)]
            model = estimator(**{**estimator_kwargs, key: best_alpha})
            # fit model
            model.fit(X_train_array, y_train_array, sample_weight=sample_weight)

            # evalute model on train data
            y_pred_train = model.predict(X_train_array) + mean_y_train
            scores_train.append(
                scoring_f(y_train_array + mean_y_train, y_pred_train + mean_y_train))

            # evaluate model on test data
            y_true = np.concatenate(y_test, axis=0)
            y_pred = model.predict(np.vstack(X_test) - mean_X_train) + mean_y_train
            if isinstance(estimator, sklm.LogisticRegression) and bins_per_trial == 1:
                y_pred_probs = model.predict_proba(
                    np.vstack(X_test) - mean_X_train)[:, 0] + mean_y_train
            else:
                y_pred_probs = None
            scores_test.append(scoring_f(y_true, y_pred))

            # save the raw prediction in the case of linear and the predicted probabilities when
            # working with logitistic regression
            for i_fold, i_global in enumerate(test_idxs_outer):
                if bins_per_trial == 1:
                    # we already computed these estimates, take from above
                    predictions[i_global] = np.array([y_pred[i_fold]])
                    if isinstance(estimator, sklm.LogisticRegression):
                        predictions_to_save[i_global] = np.array([y_pred_probs[i_fold]])
                    else:
                        predictions_to_save[i_global] = np.array([y_pred[i_fold]])
                else:
                    # we already computed these above, but after all trials were stacked;
                    # recompute per-trial
                    predictions[i_global] = model.predict(
                        X_test[i_fold] - mean_X_train) + mean_y_train
                    if isinstance(estimator, sklm.LogisticRegression):
                        predictions_to_save[i_global] = model.predict_proba(
                            X_test[i_fold] - mean_X_train)[:, 0] + mean_y_train
                    else:
                        predictions_to_save[i_global] = predictions[i_global]

            # save out other data of interest
            idxes_test.append(test_idxs_outer)
            idxes_train.append(train_idxs_outer)
            weights.append(model.coef_)
            if model.fit_intercept:
                intercepts.append(model.intercept_)
            else:
                intercepts.append(None)
            best_params.append({key: best_alpha})

    ys_true_full = np.concatenate(ys, axis=0)
    ys_pred_full = np.concatenate(predictions, axis=0)
    outdict = dict()
    outdict['scores_test_full'] = scoring_f(ys_true_full, ys_pred_full)
    outdict['scores_train'] = scores_train
    outdict['scores_test'] = scores_test
    outdict['Rsquared_test_full'] = r2_score(ys_true_full, ys_pred_full)
    if estimator == sklm.LogisticRegression:
        outdict['acc_test_full'] = accuracy_score(ys_true_full, ys_pred_full)
        outdict['balanced_acc_test_full'] = balanced_accuracy_score(ys_true_full, ys_pred_full)
    outdict['weights'] = weights
    outdict['intercepts'] = intercepts
    outdict['target'] = ys
    outdict['predictions_test'] = predictions_to_save if save_predictions else None
    outdict['regressors'] = Xs if save_binned else None
    outdict['idxes_test'] = idxes_test
    outdict['idxes_train'] = idxes_train
    outdict['best_params'] = best_params
    outdict['n_folds'] = n_folds
    if hasattr(model, 'classes_'):
        outdict['classes_'] = model.classes_

    # logging
    if verbose:
        # verbose output
        if outer_cv:
            print('Performance is only described for last outer fold \n')
        print("Possible regularization parameters over {} validation sets:".format(n_folds))
        print('{}: {}'.format(list(hyperparam_grid.keys())[0], hyperparam_grid))
        print("\nBest parameters found over {} validation sets:".format(n_folds))
        print(model.best_params_)
        print("\nAverage scores over {} validation sets:".format(n_folds))
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, model.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("\n", "Detailed scores on {} validation sets:".format(n_folds))
        for i_fold in range(n_folds):
            tscore_fold = list(
                np.round(model.cv_results_['split{}_test_score'.format(int(i_fold))], 3))
            print("perf on fold {}: {}".format(int(i_fold), tscore_fold))

        print("\n", "Detailed classification report:", "\n")
        print("The model is trained on the full (train + validation) set.")

    return outdict

