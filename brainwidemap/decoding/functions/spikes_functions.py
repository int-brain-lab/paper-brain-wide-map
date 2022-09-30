import numpy as np
import pandas as pd
from pathlib import Path
from brainbox.population.decode import get_spike_counts_in_bins
import one.alf.io as alfio
from tqdm import tqdm
from brainwidemap.decoding.functions import utils as dut
import brainbox.io.one as bbone


def get_neural_activity(df_insertions, eid, one, wideFieldImaging_dict=None, **kwargs):
    trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=['firstMovement_times'])
    trialsdf['react_times'] = trialsdf['firstMovement_times'] - trialsdf[kwargs['align_time']]
    mask = trialsdf[kwargs['align_time']].notna() & trialsdf['firstMovement_times'].notna()
    if kwargs['no_unbias']:
        mask = mask & (trialsdf.probabilityLeft != 0.5).values
    if kwargs['min_rt'] is not None:
        mask = mask & (~(trialsdf.react_times < kwargs['min_rt'])).values
    nb_trialsdf = trialsdf[mask]  # take out when mouse doesn't perform any action
    brainreg = dut.BrainRegions()
    if kwargs['merged_probes'] and wideFieldImaging_dict is None:
        across_probes = {'regions': [], 'clusters': [], 'times': [], 'qc_pass': []}
        for i_probe, (_, ins) in tqdm(enumerate(df_insertions.iterrows()), desc='Probe: ', leave=False):
            probe = ins['probe']
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            spikes = alfio.load_object(spike_sorting_path, 'spikes')
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = dut.remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            across_probes['regions'].extend(beryl_reg)
            across_probes['clusters'].extend(spikes.clusters if i_probe == 0 else
                                             (spikes.clusters + max(across_probes['clusters']) + 1))
            across_probes['times'].extend(spikes.times)
            across_probes['qc_pass'].extend(qc_pass)
        across_probes = {k: np.array(v) for k, v in across_probes.items()}
        # warnings.filterwarnings('ignore')
        if kwargs['single_region']:
            regions = [[k] for k in np.unique(across_probes['regions'])]
        else:
            regions = [np.unique(across_probes['regions'])]
        df_insertions_iterrows = pd.DataFrame.from_dict({'1': 'mergedProbes'},
                                                        orient='index',
                                                        columns=['probe']).iterrows()
    elif wideFieldImaging_dict is None:
        df_insertions_iterrows = df_insertions.iterrows()
    else:
        regions = wideFieldImaging_dict['atlas'].acronym.values
        df_insertions_iterrows = pd.DataFrame.from_dict({'1': 'mergedProbes'},
                                                        orient='index',
                                                        columns=['probe']).iterrows()

    activities = {}
    for i, ins in tqdm(df_insertions_iterrows, desc='Probe: ', leave=False):
        probe = ins['probe']
        activities_probe = {}
        if not kwargs['merged_probes']:
            spike_sorting_path = Path(ins['session_path']).joinpath(ins['spike_sorting'])
            spikes = alfio.load_object(spike_sorting_path, 'spikes')
            clusters = pd.read_parquet(spike_sorting_path.joinpath('clusters.pqt'))
            beryl_reg = dut.remap_region(clusters.atlas_id, br=brainreg)
            qc_pass = (clusters['label'] >= kwargs['qc_criteria']).values
            regions = np.unique(beryl_reg)
        # warnings.filterwarnings('ignore')
        for region in tqdm(regions, desc='Region: ', leave=False):
            if kwargs['merged_probes'] and wideFieldImaging_dict is None:
                reg_mask = np.isin(across_probes['regions'], region)
                reg_clu_ids = np.argwhere(reg_mask & across_probes['qc_pass']).flatten()
            elif wideFieldImaging_dict is None:
                reg_mask = beryl_reg == region
                reg_clu_ids = np.argwhere(reg_mask & qc_pass).flatten()
            else:
                region_labels = []
                reg_lab = wideFieldImaging_dict['atlas'][
                    wideFieldImaging_dict['atlas'].acronym == region].label.values.squeeze()
                if 'left' in kwargs['wfi_hemispheres']:
                    region_labels.append(reg_lab)
                if 'right' in kwargs['wfi_hemispheres']:
                    region_labels.append(-reg_lab)
                reg_mask = np.isin(wideFieldImaging_dict['regions'], region_labels)
                reg_clu_ids = np.argwhere(reg_mask)
            N_units = len(reg_clu_ids)
            if N_units < kwargs['min_units']:
                continue
            # or get_spike_count_in_bins
            if np.any(np.isnan(nb_trialsdf[kwargs['align_time']])):
                # if this happens, verify scrub of NaN values in all aign times before get_spike_counts_in_bins
                raise ValueError('this should not happen')
            intervals = np.vstack([nb_trialsdf[kwargs['align_time']] + kwargs['time_window'][0],
                                   nb_trialsdf[kwargs['align_time']] + kwargs['time_window'][1]]).T

            if kwargs['merged_probes'] and wideFieldImaging_dict is None:
                spikemask = np.isin(across_probes['clusters'], reg_clu_ids)
                regspikes = across_probes['times'][spikemask]
                regclu = across_probes['clusters'][spikemask]
                arg_sortedSpikeTimes = np.argsort(regspikes)
                binned, _ = get_spike_counts_in_bins(regspikes[arg_sortedSpikeTimes],
                                                     regclu[arg_sortedSpikeTimes],
                                                     intervals)
            elif wideFieldImaging_dict is None:
                spikemask = np.isin(spikes.clusters, reg_clu_ids)
                regspikes = spikes.times[spikemask]
                regclu = spikes.clusters[spikemask]
                binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
            else:
                frames_idx = wideFieldImaging_dict['timings'][kwargs['align_time']].values
                frames_idx = np.sort(
                    frames_idx[:, None] + np.arange(0, kwargs['wfi_nb_frames'], np.sign(kwargs['wfi_nb_frames'])),
                    axis=1,
                )
                binned = np.take(wideFieldImaging_dict['activity'][:, reg_mask], frames_idx, axis=0)
                binned = binned.reshape(binned.shape[0], -1).T

            activities_probe[region] = binned.T
        activities[probe] = activities_probe
    return activities # number of trials x nb bins
