# Pipeline for running BWM decoding analysis

## Installation

Follow the instruction [at the top of this repository](https://github.com/int-brain-lab/paper-brain-wide-map#readme) 
to create the base environment. On top of that environment, install behavior_models and openturns for the decoding analysis specifically:

```
git clone https://github.com/int-brain-lab/behavior_models.git
cd behavior_models
pip install -e .
pip install openturns
```

## Introduction

Decoding analysis for the BWM paper consists of regression where regressors are neuron spike counts
across trials and targets are task or behavior variables across trials.  Neurons within a given session
and region are used for a single decoding.  In order to analyze all of the BWM regions we parallelize the 
code on a slurm based cluster.  The following pipeline steps load the data, perform the regression (both for 
the real data and for a non-parametric null distribution), and save the results in a format which is 
accessible for plotting.

You must save a `settings.py` file in this directory before executing this pipeline.  The file needs to be 
in the format of `settings_template.py`.  You may copy and rename that file, making any desired settings 
changes.  This file also determines where results will be saved and the file namings for this decoding run.  
`SETTINGS_FORMAT_NAME` plays a central role in identifying a particular decoding run.  If this variable does 
not change and the decoding pipeline is run multiple times, decoding computation which has already been 
completed will not be re-computed i.e. previous computations will be skipped.  But if this variable is 
changed then all decoding will be re-run with the pipeline.  
Notably, the `SETTINGS_FORMAT_NAME` variable can be changed simply by changing the date.  
*It is recommended, therefore, that you change the date for a fresh decoding run of a given target variable.*

Additionally, you will often be submitting "*.sh" files to slurm.  The slurm settings within these files may
need to be adapted for your cluster.  The file names and directories for the "*.out" and "*.err" files will
also need to be changed to your desired names and directories. 

For an example of how to perform decoding for a single session, see the file 
`decoding_example_script.py` in this directory. The `settings.py` file does not need to be copied 
or updated to run this example.

The instructions below detail how to run a large-scale decoding analysis across the entire 
brainwide map. 

## Caching the data

We do not want to load data from ONE every time we run decoding analysis.  In particular, we are concerned 
about remotely connecting to ONE with many parralel jobs such that we overload the system.  To mitigate any 
issues, we first cache all the data we need.  Subsequent pipeline calls to ONE can then be done in the 
`local` mode.

Run the `00_data_caching.py` file to do this.  Caching takes many hours, and so should be done in parallel.  
Be careful not to do this with too many jobs so as to not overload the server.  We submit 50 jobs 
(see `N_PARA` and `para_index` in script) which is not too many. Submit these jobs using 

```
sbatch 00_slurm_data_caching.py
```

## Generate session dataframe

The cached sessions from the last step need to be aggregated into a pandas dataframe so that the many jobs 
submitted by the `04_slurm*` script do not need to re-compute the bwm session list (this has led to issues
in the past because the cache is accessed with many different workers at once).

Additionally, we use this step to save all the BiasedChoiceWorld session eids into a dataframe to be used 
by the next step.

This step is executed by the `01_generate_session_df.py` script which is submitted and run by the command, 
```
sbatch 01_slurm_generate_session_df.sh
```

## Caching imposter sessions

Imposter sessions are used to provide a set of pseudo regression targets. These pseudo regression targets are
decoded to produce a null distribution for our decoding test statistics. This step caches these targets so
that they can be accessed locally as in the previous step. The imposter sessions may be taken from ephys 
choice world or biased choice world. This is determined by the `settings.py` file parameter, i.e. whether  
`imposter_generate_from_ephys` is True of False, respectively. 
The `02_imposter_caching.py` script does this and is run in parallel (default of 500 jobs) like the last step 
with the command,

```
sbatch 02_slurm_imposter_caching.sh
```

## Generate imposter dataframe

The imposter sessions from the last step need to be aggregated into a pandas dataframe to be easily accessed 
by the rest of the pipeline. This is done by `03_generate_imposter_df.py`. This is not done is parallel, but 
nevertheless, can be summited to a slurm cluster using

```
sbatch 03_slurm_generate_imposter_df.py
```

## Decoding sessions

The regression is performed in this step by running 
```
04_decode_single_session.py X
```
where `X` is an integer which indexes across jobs.  Each sessions gets at least 1 job, so there are many 
hundreds of jobs.  These jobs are submitted in parallel using 

```
sbatch 04_slurm_decode.sh
```
The `04_decode_single_session.py` script is meant to be run on the cluster and not alone.  Thus, it is not 
recommended that you manipulate or control the `04_decode_single_session.py` input integers `X`, which 
determine the session and subset of null sessions, in order to run only a subset of the BWM datset. 
Rather, you can filter for a subset of subjects in the BWM dataset as discussed in the next section.  
Additionally, to see an example of running a single session, see `decoding_example_script.py`.
NB: some slurm clusters have a maximum number of jobs, which means you will have to submit multiple 
times.  For example, if you cannot sumbit more than 1000 jobs but require 1100, run
```
sbatch 04_slurm_decode.sh
```
where the script contains the line 
```
#SBATCH --array=1-1000
```
Then, when you are allowed to submit more jobs run the same but change the line to
```
#SBATCH --array=1001-1100
```
The total number of jobs required will be the `number of sessions * ceil(n_pseudo / n_pseudo_per_job)`.

The prefix of the filenames of output and error files found in `04_slurm_decode.sh` (lines `#SBATCH --output` 
and `#SBATCH --error`) is used for post-processing.  For example, if your error filename in 
`04_slurm_decode.sh`, excluding directory, is `ds_bwmrelease_3_.%a.err`, then you will want to remember the 
prefix `ds_bwmrelease_3_` because it is used in the next step.  Also, note the directory in which these 
files are saved and change if necessary. 

## Check decoding

Did any decoding jobs cancel (e.g. due to time limit) or did any jobs contain regression convergence 
errors? This information is in the slurm error files.  Running 
```
python 03b_check_convergence.py job_name
```
where job_name is the error file prefix used in the previous step, will read all of these files and list 
the cancellations or convergence error warnings in the command line.  Error files are found by matching the
regex `job_name+".*err"`.  If there are cancellations, read the corresponding error/output files and see 
why.  If it is due to time, simply repeat the previous step -- completed session/regions will be skipped 
and time will be used on the remaining session/regions.  This is typically necessary for wheel-speed 
decoding on my cluster which has a max time limit of 48 hours.

There is not a slurm file to submit this job because it is most useful to run on your current node, and the
outputs are printed directly in the REPL.

## Format results and save in dataframes

Dealing with the files saved for individual decoding runs can be unweildy.  Here, we read all the decoding 
results and aggregate the most important information into a pandas dataframe using `05_format_results.py`.  
This can take awhile so it is parallelized and submitted using
```
sbatch 05_slurm_format.sh
```
where the default is to parallelize 50 ways (`N_PARA=50`).  In that case 50 dataframes will be saved, and 
they are combined in the next step

## Generate summary tables

The final step of aggregating the data is to combine the dataframes from the previous step (e.g. 50) into a 
single dataframe which is useful for plotting.  The type of table saved depends on the variable being 
decoded (see `settings.py`).  Again, this can be accomplished by running
```
sbatch 06_slurm_generate_summary.sh
```

NB: wheel-speed summary table takes a lot of memory to construct, so set the slurm run to have 128GB.

## Run a subset of the BWM dataset

Running a subset of the BMW dataset can be helpful for debugging or quick analysis purposes.
You can filter for a subset of subjects in the BWM dataset by adjusting `04_slurm_decode.sh`.
Specifically, comment out this line of the file
```
python 04_decode_single_session.py $SLURM_ARRAY_TASK_ID
```
and uncomment the last line of the file.  This adds a series of subjects as inputs to
the `04_decode_single_session.py` script, and you can add or remove any number of these subjects.  The scr$
will filter for only BWM sessions of these subjects.

To see where the filtering occurs in `04_decode*.py` look for the if statement `if len(sys.argv) > 2:` 
and the comment above it.

Additionally, to see an example of running a single sessions, see `decoding_example_script.py`.


