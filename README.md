# paper-brain-wide-map
Code related to the brainwide map paper

# Contents

- [Installation](#installation)
- [Demo](#minimal-working-example-query-information-about-bwm-data)

# Installation
The installation instructions go through the step of setting up a conda environment, installing dependencies, and linking this repository source code to the environment.
This usually takes a few minutes on a fast internet connection and a consumer laptop.

## Create a new python environment (optional)

Install [Anaconda](https://www.anaconda.com/distribution/#download-section) and [git](https://git-scm.com/downloads), 
and follow their installer instructions to add each to the system path

Create new python environment
```
conda create --name ibl_bwm python=3.10
```
Activate environment
```
conda activate ibl_bwm
```

## Download and Install repository

Clone the repo 
```
git clone https://github.com/int-brain-lab/paper-brain-wide-map.git
```

Navigate to repo
```
cd paper-brain-wide-map
```

Install requirements and repo.
Note a frozen set of requirements is provided in `requirements_frozen.txt` for reference.

The installation will rely on `requirements.txt` to maintain compatibility with the latest scientific Python ecosystem.
```
pip install -e .
```

Install ibllib without dependencies
```
pip install ibllib --no-deps
```

To install additional requirements for the individual analyses, see the README files in the respective subfolders in `brainwidemap`

# Minimal working example: query information about BWM data
You can now use the following in Python after having setup ONE as described [here]()
The first connection should take a minute, subsequent connections should be faster.


```python
from brainwidemap import bwm_query, bwm_units
from one.api import ONE

one = ONE(base_url='https://openalyx.internationalbrainlab.org')
# Dataframe with info on all sessions and probes released for the BWM
bwm_df = bwm_query(one)
# Dataframe with information on all neurons used in the analyses in the BWM paper
unit_df = bwm_units(one)
unit_df.describe()
```
Should return
```
In [2]: unit_df.describe()
Out[2]: 
             depths      channels    cluster_id       amp_max  ...             z      atlas_id      axial_um    lateral_um
count  22113.000000  22113.000000  22113.000000  22113.000000  ...  22113.000000  2.211300e+04  22113.000000  22113.000000
mean    1871.847330    185.681273    314.001583      0.000356  ...     -0.003175  2.819285e+07   1871.847330     35.225388
std     1074.440921    107.449658    245.571960      0.000216  ...      0.001371  1.079593e+08   1074.440921     19.255570
min       20.000000      0.000000      0.000000      0.000075  ...     -0.007059  2.000000e+00     20.000000     11.000000
25%      920.000000     91.000000    133.000000      0.000216  ...     -0.004008  3.250000e+02    920.000000     11.000000
50%     1800.000000    179.000000    261.000000      0.000303  ...     -0.003249  6.720000e+02   1800.000000     43.000000
75%     2840.000000    282.000000    429.000000      0.000431  ...     -0.002253  1.007000e+03   2840.000000     59.000000
max     3840.000000    383.000000   1686.000000      0.004317  ...      0.000053  6.144543e+08   3840.000000     59.000000
```
For further data loading examples see `data_loading_examples.py`