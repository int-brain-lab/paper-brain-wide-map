import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ibllib.atlas import AllenAtlas
from brainwidemap.encoding.scripts.impostor_fits.summary_time_plots_perkern import atlas_variable


st.title("Examining Brain-Wide Map data results")

st.markdown("This is a work in progress.")
st.markdown(
    "Please upload a parquet file in which the index contains at least one level using `region` as a key, which will be aggregated over for statistics and plots. Initially when no file is provided, we just have a list of regions with random values (40 per region)."
)

file = st.file_uploader("Upload a parquet file", type="parquet")
atlas = AllenAtlas()

if file is None:
    uniquereg = np.unique(atlas.regions.acronym[atlas.regions.mappings["Beryl"]])
    regions = np.hstack([uniquereg] * 40)
    values = np.random.randn(len(regions))
    df = pd.DataFrame(
        values.reshape(-1, 1), index=pd.Series(regions, name="region"), columns=["test_value"]
    )
    df = df.sort_index()
else:
    df = pd.read_parquet(file)
if "region" in df.columns:
    st.text("Regions are provided in the dataframe, but are not in the index.")
    st.text("Regions will be added to the index.")
    df = df.set_index("region", append=True)
if st.checkbox("Show dataframe"):
    st.text("Here's what you uploaded:")
    st.write(df)

VALID_IDX = {
    ("region",): "Per-region",
    ("eid", "pid"): "Per-insertion",
    ("eid", "pid", "region"): "Per-insertion-region",
    ("eid", "pid", "clu_id"): "Per-unit",
    ("eid", "pid", "clu_id", "region"): "Per-unit-and-region",
}

if df.index.names in VALID_IDX:
    idxtype = VALID_IDX[df.index.names]
    st.text(f"The dataframe uses a valid combination of indices for {idxtype} statistics.")
else:
    st.text(df.index.names)

aggregators = {
    "Mean": np.mean,
    "Median": np.median,
    "Standard deviation": np.std,
    "95th Percentile": lambda x: np.quantile(x, 0.95),
    "5th Percentile": lambda x: np.quantile(x, 0.05),
    "Fraction significant": None,
}

col1, col2, col3 = st.columns(3)
with col1:
    colname = st.selectbox("Select a column to plot", df.columns)
with col2:
    aggregator = st.selectbox("Select an aggregation statistic", aggregators.keys())
with col3:
    rangetype = st.selectbox(
        "Plotting color range type",
        ["fixed", "quantile"],
        index=0 if aggregator == "Fraction significant" else 1,
    )
with st.form("plottingparams"):
    if aggregator == "Fraction significant":
        col1, col2 = st.columns(2)
        with col1:
            tailside = st.selectbox("Select tail side", ["both", "upper", "lower"])
        with col2:
            alpha = st.slider(
                "Select alpha for significance", min_value=0.0, max_value=0.1, value=0.01
            )
        if tailside == "bottom":
            agg = df.groupby("region").apply(lambda x: np.sum(x[colname] <= alpha) / len(x))
        elif tailside == "upper":
            agg = df.groupby("region").apply(lambda x: np.sum(x[colname] >= (1 - alpha)) / len(x))
        else:
            folded = 2 * (df[colname] - 0.5).abs()
            agg = folded.groupby("region").apply(lambda x: np.sum(x >= (1 - alpha) / len(x)))
    else:
        agg = df.groupby("region").agg(aggregators[aggregator])[colname]
    if rangetype == "quantile":
        qmin, qmax = st.select_slider(
            "Select color limit quantiles",
            options=np.round(np.linspace(0, 1, 20), 4),
            value=(0, 1),
        )
        vmin, vmax = np.quantile(agg, qmin), np.quantile(agg, qmax)
    elif rangetype == "fixed":
        col1, col2 = st.columns(2)
        with col1:
            vmin = st.number_input("Minimum value of colormap", value=0)
        with col2:
            vmax = st.number_input("Maximum value of colormap", value=1.0)
    submitted = st.form_submit_button("Plot")
    if submitted:
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        with st.spinner("Plotting..."):
            fig, ax, colorbar = atlas_variable(
                agg, "viridis", vmin=vmin, vmax=vmax, cbar=True, fig=fig, axes=ax, atlas=atlas
            )
        st.write(fig)
