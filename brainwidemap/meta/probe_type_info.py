# Get unique probe model across all insertions

# Author: Gaelle Chapuis
from brainwidemap import bwm_query
from one.api import ONE
import numpy as np

one = ONE()
bwm_df = bwm_query(one, return_details=True)


# Add empty colummn
bwm_df["probe_model"] = ""

ins_all = list()
model_all = list()
for row in bwm_df.iterrows():
    row_number = row[0]
    val_df = row[1]
    ins = one.alyx.rest('insertions', 'list', id= val_df['pid'])[0]
    ins_all.append(ins)
    model_all.append(ins['model'])

    bwm_df.loc[bwm_df.index[row_number], 'probe_model'] = ins['model']

print(np.unique(model_all))

count_model = bwm_df.groupby(by='probe_model').count()

print(count_model['pid'])