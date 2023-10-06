This directory contains the settings files for comparing against the prior paper decoding results.  
Comparison is necessary because the prior paper finds many more significant regions than BWM block decoding.
Through a progression of decoding runs that add/remove settings, we find that the major contribution to this
difference is the first 90 trials and linear regression.
This directory contains the settings files that implement this progression of decoding runs.  
In the progression of runs, single settings are added/removed.  See the four settings files ``*.py`` 
for details.

*Note that the settings files are sometimes copied from older settings files, and there may be missing or
additional settings that cause errors.  If this occurs just add or remove settings to match
`settings_template.py`*

This directory also contains a text file from Charles Findling that contains settings for the prior paper.
