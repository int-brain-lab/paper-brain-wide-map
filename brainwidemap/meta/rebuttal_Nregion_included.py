
from brainwidemap import bwm_units
from one.api import ONE
import numpy as np
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import pandas as pd

one = ONE()
ba = AllenAtlas()
br = BrainRegions()

units_df = bwm_units(one)

# Number of region in total in canonical set
canonical_set = units_df['Beryl'].unique()

# Number of regions in total in Beryl atlas
total_id = np.unique(br.mappings['Beryl'])
total_set = br.index2acronym(total_id, mapping='Beryl')

# Set of brain regions missing
missing_set = set(total_set) - set(canonical_set)

# Regions we removed
reg_remove = ['MOB', 'AOB', 'AOBgr', 'onl', 'AOBmi', 'void', 'root']

# Set of brain region remaining to compute area analysis
area_set = missing_set - set(reg_remove)

# Remap to Beryl and compute area volumes
# ba.label = ba.regions.mappings['Beryl-lr'][ba.label]  # Added from Mayo to remap to Beryl
ba.compute_regions_volume()  # Note: the mapping applied is Allen, not Beryl

# Prepare data frame with brain region volume
df = pd.DataFrame()
df['area'] = ba.regions.volume
df['acronyms'] = ba.regions.acronym
# # Remap DF to Beryl
df['aids_allen'] = br.acronym2id(df['acronyms'], mapping='Allen')
df['aids_beryl'] = br.remap(df['aids_allen'], source_map='Allen', target_map='Beryl')
# Translate acronyms to Beryl map
df['Beryl'] = br.id2acronym(df['aids_beryl'], mapping='Beryl')
# Drop unnecessary columns
df = df.drop(columns=['aids_allen', 'aids_beryl'])

# Sum the area as per Beryl parcellation
df2 = df.groupby('Beryl').agg(
    area=pd.NamedAgg(column="area", aggfunc="sum")
    )

# Keep only area-set regions
df_regions = df2.loc[df2.index.isin(area_set)]
# Average STD area
df_regions['area'].mean()