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

## Query sessions for BWM analyses
You can now use the following in Python (note that ONE needs to be set up)
```python
from brainwidemap import bwm_query
from one.api import ONE

one = ONE()
bwm_df = bwm_query(one)
```