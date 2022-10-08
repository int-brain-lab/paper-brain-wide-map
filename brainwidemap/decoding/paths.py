import os
from pathlib import Path

username = os.environ["USER"]  # os.getlogin()
if username == 'mattw':
    out_dir = Path('/media/mattw/ibl/')
elif username == 'mw3323':
    out_dir = Path('/home/mw3323/ibl/')
elif username == 'findling':
    out_dir = Path('/home/users/f/findling/scratch/ibl/prior-localization/braindelphi')
elif username == 'csmfindling':
    out_dir = Path('/Users/csmfindling/Documents/Postdoc-Geneva/IBL/code/prior-localization/braindelphi')
elif username == 'hubertf':
    out_dir = Path('/home/users/h/hubertf/scratch/')
elif username == 'bensonb':
    out_dir = Path('/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/results')


# path to user-specific settings file
# SETTINGS_PATH = out_dir.joinpath('decoding', 'settings.yaml')

# store cached data for simpler loading
CACHE_PATH = out_dir.joinpath('cache')

# store neural decoding models
FIT_PATH = out_dir.joinpath('decoding', 'results', 'neural')

# store behavioral models
BEH_MOD_PATH = out_dir.joinpath('decoding', 'results', 'behavioral')

# store imposter session data used for creating null distributions
IMPOSTER_SESSION_PATH = out_dir.joinpath('decoding')

# store imposter session data used for creating null distributions
INTER_INDIVIDUAL_PATH = out_dir.joinpath('decoding')
