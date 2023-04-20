# Encoding models of the brain-wide map

This section of the repo is dedicated to the specification, fitting, and analysis of the encoding models fit on a single-neuron basis to the IBL brain wide map.

The pipelines used for the processing of the entire dataset are located in the `pipelines/` folder, and should be run in sequence for reproduction of the results in the paper.

## Installation

There are several steps that are necessary to get the code in this section of the repo to run. All of these installation instructions should be run within the parent IBL environment specified [in this GitHub repo](https://github.com/int-brain-lab/iblenv), on top of an existing IBLenv environment.

First and foremost the user must install the `neurencoding` package upon which the models are built. This can be done by simply running `pip install neurencoding` within the IBL environment.

Users must also adjust the paths for storage in the `paper-brain-wide-map/brainwidemap/encoding/params.py` file. These are the paths where the code will store cached results, `GLM_CACHE`, and the fit results for the brain-wide map, `GLM_FIT_PATH`. As long as the top-level directory that is specified already exists the code will handle subfolder generation etc.

If you would like to run the full pipeline for data analyses, there are additional required packages and steps, as specified in the [pipelines section](#pipelines) of this document.

## Fitting a single session

If you'd like to begin by fitting a single session, the `scripts/fit_example_session.ipynb` notebook will provide a good jumping off point for understanding how the leave-one-out analysis works, and the results it produces.

## Overview of functionality

### Pipelines

The `pipelines/` folder, as mentioned, contains all of the scripts necessary for running the full fit of the data on a SLURM-based cluster. This requires a cluster which uses both SLURM and allows the user to load in modules for [Singularity](https://docs.sylabs.io/guides/3.7/user-guide/).

Singularity is a containerization system, much like Docker (and in fact can build containers from docker images), which is intended for use in HPC environments. This reduces the load on network file systems that are used in HPC environments and can often bottleneck complicated python environments such as the one in `iblenv`, as was the case when these analyses were run. Singularity does this in a manner where, unlike Docker, root permissions are not necessary to run a container. This is useful in a limited-permissions environment like those found on mose HPC clusters.

The `Dockerfile`, found in this folder, specifies the docker image used to run these analyses. For convenience, a built image of this container can be found on DockerHub as well under `bdgercek/iblcore` which contains all the necessary packages (including `neurencoding`) to run the pipeline. From that package a simple `singularity build ./<your_image_name>.sif docker://bdgercek/iblcore` will suffice to compile a local singularity image. This image will then be passed to the scripts in `pipelines/` which must be modified with the path to said image. 

Be warned that the image is quite large due to the heavy dependencies of `iblenv`, often reaching 3GB. This is because it is a swiss-army-knife container that also includes `pytorch` and CUDA. In the future I will likely upload a pared-down version built on simple ubuntu, but even that will still be large because of the python libraries.

Users will also have to install the `dask`, `distributed`, and `dask-jobqueue` packages for the SLURM pipelines to work properly.

### Scripts

The `scripts/` folder contains small scripts that either run plotting or simple analyses for use in verifying and expanding the model. The example of a single-session fit is contained here, `fit_example_session.ipynb`. This also contains scripts for running GLMs on only the inter-trial-interval, as well as simple linear regression to predict the block identity (`single_unit_iti_{glm | pleft_regression}.py`). `twocond_plots` will compare the PSTH and predictions of the model for a set of example units that are selected for selectivity to each regressor. `xcorr_events_spikes.py` examines the cross-correlation of all units in the BWM with various even timings, and produces plots that were used to select kernel lengths.

### Cluster worker

`cluster_worker.py` implements a mother script for cluster workers to process individual probe insertions. This relies on a cached dataset, produced using the `pipelines/01_cache_regressors.py` script, as well as several files specifying the identity and parameters of a cached dataset and the parameters of the current run of the model.

### Design matrix

`design.py` specifies the design matrix used in the model. Arguably the most critical component of the single-unit encoding models, this design matrix describes the regressors which we use to predict single-unit firing.

### Fitting methods

`fit.py` contains the core functions for fitting neural data to the design matrix produced by `design.py`, particularly using the leave-one-out methodology which is a sub-method of stepwise regression.

### GLM prediction and analysis

`glm_predict.py` contains utilities for comparing peri-event-time-histograms (PETHs) of real data against those predicted by the model, aligned to arbitrary events on an arbitrary subset of trials

### Synthetic data testing and generation

`synth.py` is a set of functions for generating data using the same assumptions underlying the encoding models and then fitting that data using the model to examine recovery of generative parameters. In effect this means taking a set of parameters fit by the model, and producing spike trains from those parameters to simulate a cell in which the true generative process is known exactly. Then that synthetic spike train can be used to fit one again, and see how well the model can capture a perfectly known output.

### Time series alignement

`timeseries.py` contains basic utilities used for the generation of the trials dataframe which governs design matrix generation. Particularly for the resampling of wheel position down to a fixed sampling interval, which is then used to compute velocity.

### Utilities

`utils.py` is a catch-all for functions that have broad use in the analysis. `load_trials_df` in particular will load the dataframe of trial events, values, and timings for a given session and append (optionally) the wheel velocity or speed for that session.

