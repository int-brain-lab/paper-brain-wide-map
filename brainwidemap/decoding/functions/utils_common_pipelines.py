# Standard library
import hashlib
import logging
import os
import pickle
import re
from datetime import datetime as dt
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd

# IBL libraries
import brainbox.io.one as bbone
import brainbox.metrics.single_units as bbqc
from ibllib.dsp.smooth import smooth_interpolate_savgol
from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

_logger = logging.getLogger('braindelphi')


def load_behavior(eid, target, one=None):
    """Load raw behavior traces and their timestamps.

    Parameters
    ----------
    eid : str
    target : str
    one : ONE object

    Returns
    -------
    dict
        keys are 'times', 'values', and 'skip' (meaning error loading data)

    """

    one = ONE() if one is None else one

    target_times = None
    target_vals = None
    skip_session = False

    try:

        if target == 'wheel-vel' or target == 'wheel-speed':
            target_times, _, target_vals, _ = load_wheel_data(one, eid)
            if target == 'wheel-speed':
                target_vals = np.abs(target_vals)
        elif target == 'pupil':
            target_times, target_vals = load_pupil_data(one, eid, snr_thresh=5)
        elif target.find('paw') > -1:  # '[l/r]-paw-[pos/vel/speed]'
            target_times, target_vals = load_paw_data(
                one, eid, paw=target.split('-')[0], kind=target.split('-')[2])
        elif target == 'l-whisker-me' or target == 'r-whisker-me':
            target_times, target_vals = load_whisker_data(
                one, eid, view=target.split('-')[0])
        else:
            raise NotImplementedError('%s is an invalid decoding target' % target)

    except TimestampError:
        msg = '%s timestamps/data mismatch' % target
        _logger.exception(msg)
        skip_session = True
    except ALFObjectNotFound:
        msg = 'alf object for %s not found' % target
        _logger.exception(msg)
        skip_session = True
    except QCError:
        msg = '%s trace did not pass QC' % target
        _logger.exception(msg)
        skip_session = True
    except ValueError:
        msg = 'not enough good pupil points'
        _logger.exception(msg)
        skip_session = True

    if not skip_session:
        if target_times is None:
            msg = 'no %s times' % target
            _logger.info(msg)
            skip_session = True
        if target_vals is None:
            msg = 'no %s vals' % target
            _logger.info(msg)
            skip_session = True

    return {'times': target_times, 'values': target_vals, 'skip': skip_session}


def load_wheel_data(one, eid, sampling_rate=1000):
    from brainbox.behavior.wheel import interpolate_position, velocity_smoothed
    # wheel_pos_og = one.load(eid, dataset_types='wheel.position')[0]
    # wheel_times_og = one.load(eid, dataset_types='wheel.timestamps')[0]
    wheel_pos_og = one.load_dataset(eid, '_ibl_wheel.position.npy')
    wheel_times_og = one.load_dataset(eid, '_ibl_wheel.timestamps.npy')
    if wheel_times_og.shape[0] != wheel_pos_og.shape[0]:
        raise TimestampError
    # resample the wheel position and compute velocity, acceleration
    wheel_pos, wheel_times = interpolate_position(wheel_times_og, wheel_pos_og, freq=sampling_rate)
    wheel_vel, wheel_acc = velocity_smoothed(wheel_pos, sampling_rate)
    return wheel_times, wheel_pos, wheel_vel, wheel_acc


def load_pupil_data(one, eid, view='left', snr_thresh=5):

    from brainbox.behavior.dlc import get_pupil_diameter

    assert view == 'left', 'Pupil diameter computation only implemented for left pupil'

    times, xys = get_dlc_traces(one, eid, view='left')
    if xys is None:
        return None, None

    diam0 = get_pupil_diameter(xys, view='left')
    if times.shape[0] != diam0.shape[0]:
        raise TimestampError
    pupil_diam = smooth_interpolate_savgol(diam0, window=31, order=3, interp_kind='linear')
    good_idxs = np.where(~np.isnan(diam0) & ~np.isnan(pupil_diam))[0]
    snr = np.var(pupil_diam[good_idxs]) / np.var(diam0[good_idxs] - pupil_diam[good_idxs])
    if snr < snr_thresh:
        raise QCError('Bad SNR value (%1.2f)' % snr)

    return times, pupil_diam


def load_paw_data(
        one, eid, paw, kind, smooth=True, frac_thresh=0.9, jump_thresh_px=200, n_jump_thresh=50):

    if paw == 'l':
        view = 'left'
    else:
        view = 'right'

    times, paw_markers_all, frac_present, jumps = load_paw_markers(
        one, eid, view=view, smooth=smooth, jump_thresh_px=jump_thresh_px)
    if times is None or paw_markers_all is None:
        return None, None

    # note: the *right* paw returned by the dlc dataframe is always the paw closest to the camera:
    # the paws are labeled according to the viewer's sense of L/R
    # the left view is the base view; the right view is flipped to match the left view
    # the right view (vid and markers) is flipped back, but the DLC bodypart names stay the same

    # perform QC checks
    if frac_present['r_x'] < frac_thresh:
        raise QCError(
            'fraction of present markers (%1.2f) is below threshold (%1.2f)' %
            (frac_present['r_x'], frac_thresh))
    if jumps['r'] > n_jump_thresh:
        raise QCError(
            'number of large jumps (%i) is above threshold (%i)' % (jumps['r'], n_jump_thresh))

    # process; compute position/velocity/speed
    paw_markers = np.hstack(
        [paw_markers_all['paw_r_x'][:, None], paw_markers_all['paw_r_y'][:, None]])

    if kind == 'pos':
        # x-y coordinates; downstream functions can handle this
        paw_data = paw_markers
    elif kind == 'vel':
        # x-y coordinates; downstream functions can handle this
        paw_data = np.concatenate([[[0, 0]], np.diff(paw_markers, axis=0)])
    elif kind == 'speed':
        diffs = np.sqrt(np.sum(np.square(np.diff(paw_markers, axis=0)), axis=1))
        paw_data = np.concatenate([[0], diffs])
    else:
        raise NotImplementedError(
            '%s is not a valid kind of paw data; must be "pos", "vel" or "speed"' % kind)

    return times, paw_data


def load_paw_markers(one, eid, view='left', smooth=True, jump_thresh_px=200):

    times, xys = get_dlc_traces(one, eid, view=view)
    if times is None or xys is None:
        return None, None, None, None

    # separate data
    paw_r_x = xys['paw_r'][:, 0]
    paw_r_y = xys['paw_r'][:, 1]
    paw_l_x = xys['paw_l'][:, 0]
    paw_l_y = xys['paw_l'][:, 1]

    if times.shape[0] != paw_l_y.shape[0]:
        raise TimestampError

    # compute fraction of time points present
    frac_r_x = np.sum(~np.isnan(paw_r_x)) / paw_r_x.shape[0]
    frac_r_y = np.sum(~np.isnan(paw_r_y)) / paw_r_y.shape[0]
    frac_l_x = np.sum(~np.isnan(paw_l_x)) / paw_l_x.shape[0]
    frac_l_y = np.sum(~np.isnan(paw_l_y)) / paw_l_y.shape[0]

    # compute number of large jumps
    diffs_r = np.concatenate([[[0, 0]], np.diff(xys['paw_r'], axis=0)])
    jumps_r = np.sqrt(np.sum(np.square(diffs_r), axis=1))
    n_jumps_r = np.sum(jumps_r > jump_thresh_px)
    diffs_l = np.concatenate([[[0, 0]], np.diff(xys['paw_l'], axis=0)])
    jumps_l = np.sqrt(np.sum(np.square(diffs_l), axis=1))
    n_jumps_l = np.sum(jumps_l > jump_thresh_px)

    if smooth:
        if view == 'left':
            ws = 7
        else:
            ws = 17
        paw_r_x = smooth_interpolate_savgol(paw_r_x, window=ws)
        paw_r_y = smooth_interpolate_savgol(paw_r_y, window=ws)
        paw_l_x = smooth_interpolate_savgol(paw_l_x, window=ws)
        paw_l_y = smooth_interpolate_savgol(paw_l_y, window=ws)
    else:
        raise NotImplementedError("Need to implement interpolation w/o smoothing")

    markers = {'paw_l_x': paw_l_x, 'paw_r_x': paw_r_x, 'paw_l_y': paw_l_y, 'paw_r_y': paw_r_y}
    fracs = {'l_x': frac_l_x, 'r_x': frac_r_x, 'l_y': frac_l_y, 'r_y': frac_r_y}
    jumps = {'l': n_jumps_l, 'r': n_jumps_r}

    return times, markers, fracs, jumps


def load_whisker_data(one, eid, view):
    if view == 'l':
        view = 'left'
    elif view == 'r':
        view = 'right'
    else:
        raise NotImplementedError

    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        me = one.load_dataset(eid, '%sCamera.ROIMotionEnergy.npy' % view)
    except:
        msg = 'whisker ME data not available'
        _logger.exception(msg)
        return None, None

    if times.shape[0] != me.shape[0]:
        raise TimestampError

    return times, me


def get_dlc_traces(one, eid, view, likelihood_thresh=0.9):
    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except:
        msg = 'not all dlc data available'
        _logger.exception(msg)
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs


class QCError(Exception):
    pass


class TimestampError(Exception):
    pass


def _hash_dict(d):
    hasher = hashlib.md5()
    sortkeys = sorted(d.keys())
    for k in sortkeys:
        v = d[k]
        if type(v) == np.ndarray:
            hasher.update(v)
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            hasher.update(v.to_string().encode())
        else:
            try:
                hasher.update(v)
            except Exception:
                _logger.warning(f'Key {k} was not able to be hashed. May lead to failure to update'
                                ' in cached files if something was changed.')
    return hasher.hexdigest()
