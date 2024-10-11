'''

Functions to calculate cross validated spike-triggered averages using regression models

'''

import numpy as np

# TODO: should one be first as with load_good_units?
def load_passive_rf_map(eid, normalize=True, normalize_range=(-1,1), one=None):
    """
    For a given eid load in the passive receptive field mapping protocol data

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    normalize : [bool]
        Return the receptive field map frames normalized to unit magntitude centered on baseline
    one : oneibl.one.OneAlyx, optional
        An instance of ONE (may be in 'local' - offline - mode)

    Returns
    -------
    one.alf.io.AlfBunch
        Passive receptive field mapping data
    """
    one = one or ONE()

    # Load in the receptive field mapping data
    rf_map = one.load_object(eid, obj='passiveRFM', collection='alf')
    frames = np.fromfile(one.load_dataset(eid, '_iblrig_RFMapStim.raw.bin',
                                          collection='raw_passive_data'), dtype="uint8")
    # set receptive field meta-data TODO: verify!!
    rf_map['x_dim'], rf_map['y_dim'] = 15, 15
    # TODO: provide visual angle locations!
    rf_map['frame_rate'] = 60  # Hz
    frames = np.transpose(np.reshape(frames, [rf_map['y_dim'], rf_map['x_dim'], -1], order="F"), [2, 1, 0])
    if normalize:
        frames = frames.astype('float')
        frames = frames / 255  # data is uint8, 0, 128, 255
        frames = (frames - normalize_range[0]) / (normalize_range[1] - normalize_range[0])

    rf_map['frames'] = frames

    return rf_map


