"""
Script to use new neuralGLM object from brainbox rather than complicated matlab calls

Berk, May 2020
"""

# Standard library
import argparse
import os
import pickle
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd

# Brainwide repo imports
from brainwidemap.encoding.design import generate_design
from brainwidemap.encoding.fit import fit_stepwise, fit_impostor
from brainwidemap.encoding.params import GLM_FIT_PATH


def filter_nan(trialsdf):
    target_cols = ['stimOn_times', 'feedback_times', 'firstMovement_times']
    mask = ~np.any(np.isnan(trialsdf[target_cols]), axis=1)
    return trialsdf[mask]


def get_cached_regressors(fpath):
    with open(fpath, 'rb') as fo:
        d = pickle.load(fo)
    return d['trialsdf'], d['spk_times'], d['spk_clu'], d['clu_regions'], d['clu_df']


def _create_sub_sess_path(parent, subject, session):
    subpath = Path(parent).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session)
    if not sesspath.exists():
        os.mkdir(sesspath)
    return sesspath


def save_stepwise(subject, session_id, fitout, params, probes, input_fn, clu_reg, clu_df, fitdate):
    sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    fn = sesspath.joinpath(f'{fitdate}_stepwise_regression.pkl')
    outdict = {
        'params': params,
        'probes': probes,
        'model_input_fn': input_fn,
        'clu_regions': clu_reg,
        'clu_df': clu_df,
    }
    outdict.update(fitout)
    with open(fn, 'wb') as fw:
        pickle.dump(outdict, fw)
    return fn


def save_impostor(subject, session_id, sessfit, nullfits, params, probes, input_fn, clu_reg,
                  clu_df, fitdate):
    sesspath = _create_sub_sess_path(GLM_FIT_PATH, subject, session_id)
    fn = sesspath.joinpath(f'{fitdate}_impostor_regression.pkl')
    outdict = {
        'params': params,
        'probes': probes,
        'model_input_fn': input_fn,
        'clu_regions': clu_reg,
        'clu_df': clu_df,
        'fitdata': sessfit,
        'nullfits': nullfits,
    }
    with open(fn, 'wb') as fw:
        pickle.dump(outdict, fw)
    return fn


def fit_save_inputs(
    subject,
    eid,
    probes,
    eidfn,
    params,
    t_before,
    fitdate,
    impostors,
    impostor_path=None,
):
    stdf, sspkt, sspkclu, sclureg, scluqc = get_cached_regressors(eidfn)
    stdf_nona = filter_nan(stdf)
    sessprior = stdf_nona['probabilityLeft']
    sessdesign = generate_design(stdf_nona, sessprior, t_before, **params)
    if not impostors:
        sessfit = fit_stepwise(sessdesign, sspkt, sspkclu, **params)
        outputfn = save_stepwise(subject, eid, sessfit, params, probes, eidfn, sclureg, scluqc,
                                 fitdate)
    else:
        impdf = filter_nan(pd.read_pickle(impostor_path))
        sessfit, nullfits = fit_impostor(sessdesign,
                                         impdf,
                                         sspkt,
                                         sspkclu,
                                         t_before=t_before,
                                         **params)
        outputfn = save_impostor(subject, eid, sessfit, nullfits, params, probes, eidfn, sclureg,
                                 scluqc, fitdate)
    return outputfn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster GLM fitter')
    parser.add_argument('datafile',
                        type=Path,
                        help='Input file (parquet pandas df) \
                        containing inputs to each worker')
    parser.add_argument('paramsfile', type=Path, help='Parameters for model fitting for worker')
    parser.add_argument('index',
                        type=int,
                        help='Index in inputfile for this worker to '
                        'process/save')
    parser.add_argument('fitdate', help='Date of fit for output file')
    parser.add_argument('--impostor_path', type=Path, help='Path to main impostor df file')
    args = parser.parse_args()

    with open(args.datafile, 'rb') as fo:
        dataset = pickle.load(fo)
    with open(args.paramsfile, 'rb') as fo:
        params = pickle.load(fo)
    t_before = dataset['params']['t_before']
    dataset_fns = dataset['dataset_filenames']

    subject, eid, probes, metafn, eidfn = dataset_fns.loc[args.index]

    outputfn = fit_save_inputs(subject,
                               eid,
                               probes,
                               eidfn,
                               params,
                               t_before,
                               args.fitdate,
                               params['impostor'],
                               impostor_path=args.impostor_path)
    print('Fitting completed successfully!')
    print(outputfn)
