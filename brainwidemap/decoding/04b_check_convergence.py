import os
import re
import sys
from brainwidemap.decoding.settings import SLURM_DIR


job_name = sys.argv[1]

fs = [f for f in os.listdir(SLURM_DIR) if re.match(job_name + ".*err", f)]
fs_out = [f for f in os.listdir(SLURM_DIR) if re.match(job_name + ".*out", f)]
print(f'found {len(fs)} matching error files in SLURM_DIR')
print(f'found {len(fs_out)} matching output files in SLURM_DIR')
conv_warn_files = []
cancel_files = []
for f in fs:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n", "")
        if re.match(".*ConvergenceWarning.*", s):
            conv_warn_files.append(f)
        if re.match(".*CANCELLED.*", s):
            cancel_files.append(f)

non_success_files = []
for f in fs_out:
    with open(SLURM_DIR.joinpath(f), "r") as fo:
        s = fo.read().replace("\n","")
        if not (re.match(".*Job successful.*", s) or re.match(".*ended because job counter.*", s)):
            non_success_files.append(f)

print("Convergence warning files:")
print('\n'.join(conv_warn_files))
print()
print("Cancelled files:")
print('\n'.join(cancel_files))
print()
print("Non-successful files:")
print('\n'.join(non_success_files))
