import os
from pathlib import Path
import pickle


def get_save_path(
        pseudo_id, subject, eid, neural_dtype, probe, region, output_path, time_window, date,
        target, add_to_saving_path):
    subjectfolder = Path(output_path).joinpath(neural_dtype, subject)
    eidfolder = subjectfolder.joinpath(eid)
    probefolder = eidfolder.joinpath(probe)
    start_tw, end_tw = time_window
    fn = '_'.join([date, region, 'target', target,
                   'timeWindow', str(start_tw).replace('.', '_'), str(end_tw).replace('.', '_'),
                   'pseudo_id', str(pseudo_id), add_to_saving_path]) + '.pkl'
    save_path = probefolder.joinpath(fn)
    return save_path


def save_region_results(fit_result, pseudo_id, subject, eid, probe, region, n_units, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    outdict = {
        'fit': fit_result, 'pseudo_id': pseudo_id, 'subject': subject, 'eid': eid, 'probe': probe,
        'region': region, 'N_units': n_units
    }
    fw = open(save_path, 'wb')
    pickle.dump(outdict, fw)
    fw.close()
    return save_path
