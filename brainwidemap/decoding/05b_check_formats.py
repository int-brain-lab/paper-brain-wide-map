import os
import re
import sys
from brainwidemap.decoding.settings import SLURM_DIR


job_name = sys.argv[1]

fs = [f for f in os.listdir(SLURM_DIR) if re.match(job_name + ".*err", f)]
fs_out = [f for f in os.listdir(SLURM_DIR) if re.match(job_name + ".*out", f)]

print(f'found {len(fs_out)} matching output files in SLURM_DIR')

non_success_files = []
for f in fs_out:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n","")
        if not (re.match(".*metadata saved.*", s)):
            non_success_files.append(f)

print("Non-successful files:")
print('\n'.join(non_success_files))
