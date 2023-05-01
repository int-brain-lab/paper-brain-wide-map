import os
import re
import sys
from one.api import ONE
from brainwidemap.decoding.settings import SLURM_DIR

one = ONE()
job_name = sys.argv[1]

fs = [f for f in os.listdir(SLURM_DIR) if re.match(job_name + ".*err", f)]
fs_out = [f for f in os.listdir(SLURM_DIR) if re.match(job_name + ".*out", f)]
print(f'found {len(fs)} matching error files in SLURM_DIR')
print(f'found {len(fs_out)} matching output files in SLURM_DIR')

cancel_files = []
for f in fs:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n", "")
        if re.match(".*CANCELLED.*", s):
            cancel_files.append(f)

spikesort_fail_eids = []
trial_fail_eids = []
for f in fs_out:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n","")
        matches = re.finditer("Downloading failed for spike sorting data", s)
        for match in matches:
            end_index = match.end()
            spikesort_fail_eids.append(f'eid {one.pid2eid(s[end_index+6:end_index+42])[0]}')
        matches = re.finditer("Downloading failed for trials data", s)
        for match in matches:
            end_index = match.end()
            trial_fail_eids.append(s[end_index+2:end_index+42])

print("Cancelled files:")
print('\n'.join(cancel_files))
print("Downloading failed for spike sorting:")
print('\n'.join(spikesort_fail_eids))
print("Downloading failed for trial data:")
print('\n'.join(trial_fail_eids))
