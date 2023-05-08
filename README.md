# paper-brain-wide-map
Code related to the brainwide map paper


# Installation
## Create a new python environment (optional)

Install [Anaconda](https://www.anaconda.com/distribution/#download-section) and [git](https://git-scm.com/downloads), 
and follow their installer instructions to add each to the system path

Create new python environment
```
conda create --name ibl_bwm python=3.9
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

Install requirements and repo
```
pip install -e .
```

Install ibllib without dependencies
```
pip install ibllib --no-deps
```

To install additional requirements for the individual analyses, see the README files in the respective subfolders in `brainwidemap`

## Query information about BWM data
You can now use the following in Python (note that ONE needs to be set up)
```python
from brainwidemap import bwm_query, bwm_units
from one.api import ONE

one = ONE()
# Dataframe with info on all sessions and probes released for the BWM
bwm_df = bwm_query(one)
# Dataframe with information on all neurons used in the analyses in the BWM paper
unit_df = bwm_units(one)
```

For further data loading examples see `data_loading_examples.py`