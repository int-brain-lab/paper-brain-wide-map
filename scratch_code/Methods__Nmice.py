"""
BWM paper
Methods:

N=xx adult mice (C57BL/6, male and female, obtained from either Jackson Laboratory or Charles River)
were used in this study. Mice were aged XX–XX weeks and weighed XX–XX g on the day of
electrophysiological recording.
"""
# Author: Gaelle Chapuis
from brainwidemap import bwm_query
from one.api import ONE

one = ONE()
bwm_df, insertions = bwm_query(one, return_details=True)


sub_list = list()
id_subj_all = list()
for q_i in insertions:
    subj = one.alyx.rest('subjects', 'list',
                         nickname=q_i['session_info']['subject'],
                         lab=q_i['session_info']['lab'])[0]
    id_subj = subj['id']
    if id_subj not in id_subj_all:  # Remove duplicates if any
        sub_list.append(subj)
        id_subj_all.append(id_subj)

# N SUBJECT
print(f'N subject: {len(sub_list)}')

# AGE RANGE
age_weeks = [item['age_weeks'] for item in sub_list if item['age_weeks'] != 0]
print(f'AGE RANGE: {min(age_weeks)}-{max(age_weeks)} weeks')

# WEIGHT RANGE
ref_weight = [item['reference_weight'] for item in sub_list]
print(f'WEIGHT RANGE: {min(ref_weight)}-{max(ref_weight)} g')

# Age 0
age_0 = [item['nickname'] for item in sub_list if item['age_weeks'] == 0]
# NR_0019 -> chased up Noam