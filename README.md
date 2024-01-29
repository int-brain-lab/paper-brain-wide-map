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

## Setup the paper-brain-wide-map repository

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

To install additional requirements for the individual analyses, see the README files in the respective subfolders in `brainwidemap`

# Minimal working example: query information about BWM data
You can now use the following in Python after having setup ONE as described [here](https://int-brain-lab.github.io/iblenv/notebooks_external/one_quickstart.html)
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
             depths      channels    cluster_id  ...             z      atlas_id      axial_um    lateral_um
count  31344.000000  31344.000000  31344.000000  ...  31344.000000  3.134400e+04  31344.000000  31344.000000
mean    1872.586779    185.753765    378.037296  ...     -0.003130  2.301535e+07   1872.586779     35.283818
std     1070.074064    107.010005    321.481276  ...      0.001366  9.726513e+07   1070.074064     19.206001
min       20.000000      0.000000      0.000000  ...     -0.007059  2.000000e+00     20.000000     11.000000
25%      935.000000     91.750000    147.000000  ...     -0.003970  2.660000e+02    935.000000     11.000000
50%     1800.000000    179.000000    293.000000  ...     -0.003206  6.720000e+02   1800.000000     43.000000
75%     2840.000000    282.000000    500.000000  ...     -0.002171  9.880000e+02   2840.000000     59.000000
max     3840.000000    383.000000   1829.000000  ...      0.000053  6.144543e+08   3840.000000     59.000000


```
For further data loading examples see `data_loading_examples.py`

# Reproducing a basic set of results
The following notebook shows a simple example of how to reproduce basic parts of the analysis on a subset of the data.
https://colab.research.google.com/drive/1V6Cgi8vsKz0I3BOkFuq6lULb9we82vYr