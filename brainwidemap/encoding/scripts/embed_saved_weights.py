# %%
import cuml
from ibllib.atlas import BrainRegions
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import cache
import sklearn.preprocessing as pp
import sklearn.manifold as skma


@cache
def parentreg(region, level=3):
    try:
        return br.ancestors(br.acronym2id(region))["name"][level]
    except IndexError:
        return br.name[br.acronym2index(region)[1][0][0]]


@cache
def regcolor(region):
    return br.rgba[br.acronym2index(region)[1]][0, 0] / 255


def regionscatter(x, y, data, **kwargs):
    return sns.scatterplot(
        data.query("level6 in @level6_interesting_regions"), x=x, y=y, hue="level6", **kwargs
    )


COVARS = [
    "stimonR",
    "stimonL",
    "correct",
    "incorrect",
    "fmoveR",
    "fmoveL",
    "pLeft",
    "pLeft_tr",
    "wheel",
]

covpairs = {
    "stim": ["stimonR", "stimonL"],
    "fmove": ["fmoveR", "fmoveL", "wheel"],
    "feedback": ["correct", "incorrect"],
    "pLeft": ["pLeft", "pLeft_tr"],
    "all": COVARS,
}

level6_interesting_regions = [
    "Somatomotor areas",
    "Orbital area" "Anterior cingulate area",
    "Visual areas",
    "Hippocampal region",
    "Caudoputamen",
]

# %%
fitdata = pd.read_pickle("/home/berk/2023-10-02_glm_fit.pkl")
br = BrainRegions()
weights = fitdata["fit_weights"].query("region != 'root' & region != 'void'").copy()
weights["parent"] = weights.region.apply(parentreg)
weights["level6"] = weights.region.apply(parentreg, level=6)
weights["regcolor"] = weights.region.apply(regcolor)
mask = weights["parent"].apply(type) == str
weights = weights[mask]

scaled_weights = weights.copy()
scaled_weights.loc[:, "stimonR_0":"wheel_2"] = pp.power_transform(
    scaled_weights.loc[:, "stimonR_0":"wheel_2"],
)


methods = {
    "spectr_umap": cuml.manifold.UMAP(n_components=2),
    "lown_umap": cuml.manifold.UMAP(n_components=2, n_neighbors=5),
    "highn_umap": cuml.manifold.UMAP(n_components=2, n_neighbors=100),
    "pca": cuml.PCA(n_components=2),  
}
# %%
# First single-variable embeddings
embeddings = []


def create_emb_df(weights, scaled_weights, methname, method, cov, columns, scaling):
    basedata = weights if not scaling else scaled_weights
    emb = method.fit_transform(basedata[columns].values)
    embdf = pd.DataFrame(emb, index=basedata.index, columns=["Dim 1", "Dim 2"])
    embdf["method"] = methname
    embdf["covariate"] = cov
    embdf["uniform_scaled"] = scaling
    embdf["parent"] = basedata["parent"]
    embdf["level6"] = basedata["level6"]
    embdf["region"] = basedata["region"]
    embdf["regcolor"] = basedata["regcolor"]
    return embdf


for methname, method in methods.items():
    for cov in COVARS:
        columns = weights.columns[weights.columns.str.match(cov)]
        if len(columns) <= 2:
            continue
        for scaling in (True, False):
            embdf = create_emb_df(weights, scaled_weights, methname, method, cov, columns, scaling)
            embeddings.append(embdf)
embeddings = pd.concat(embeddings)
# %%
for cov in COVARS:
    for scaling in (True, False):
        fgdata = embeddings.query(
            "covariate == @cov & uniform_scaled == @scaling &"
            " level6 in @level6_interesting_regions"
        )
        if len(fgdata) < 5:
            continue
        fg = sns.FacetGrid(
            fgdata,
            row="method",
            col="level6",
            sharex="row",
            sharey="row",
        )
        fg.map(
            sns.histplot,
            "Dim 1",
            "Dim 2",
            bins=25,
        )
        fg.add_legend()
        fg.set_titles("{col_name}\n{row_name}")
        datatype_folder = "raw_weights" if not scaling else "normaldist_rescale"
        fg.savefig(
            Path("~/Documents/Projects/results/glms/").expanduser()
            / "regions_of_interest"
            / datatype_folder
            / f"{cov}_weight_embedding_{scaling * 'rescaled_data_'}"
            "bestmethods_interestingregions.png",
            dpi=300,
        )
        plt.close()
# %%
embeddings = []
for methname, method in methods.items():
    for pname, pair in covpairs.items():
        columns = weights.columns[
            weights.columns.str.match(pair[0]) | weights.columns.str.match(pair[1])
        ]
        if len(columns) <= 2:
            continue
        for scaling in (True, False):
            embdf = create_emb_df(weights, scaled_weights, methname, method, pname, columns, scaling)
            embeddings.append(embdf)
embeddings = pd.concat(embeddings)

# %%
for pair in covpairs:
    for scaling in (True, False):
        fgdata = embeddings.query(
            "covariate == @pair & uniform_scaled == @scaling &"
            " level6 in @level6_interesting_regions"
        )
        if len(fgdata) < 5:
            continue
        fg = sns.FacetGrid(
            fgdata,
            row="method",
            col="level6",
            sharex="row",
            sharey="row",
        )
        fg.map(
            sns.histplot,
            "Dim 1",
            "Dim 2",
            bins=25,
        )
        fg.add_legend()
        fg.set_titles("{col_name}\n{row_name}")
        datatype_folder = "raw_weights" if not scaling else "normaldist_rescale"
        fg.savefig(
            Path("~/Documents/Projects/results/glms/").expanduser()
            / "regions_of_interest"
            / datatype_folder
            / f"{pair}_pair_weight_embedding_{scaling * 'rescaled_data_'}"
            "bestmethods_interestingregions.png",
            dpi=300,
        )
        plt.close()
# %%
