# Pipeline for running BWM decoding analysis

## Introduction

Decoding analysis for the BWM paper consists of regression where regressors are neuron spike counts
across trials and targets are task or behavior variables across trials.  Neurons within a given session
and region are used for a single decoding.  In order to analyze all of the BWM regions we parallelize the 
code on a slurm based cluster.  The following pipeline steps load the data, perform the regression (both for 
the real data and for a non-parametric null distribution), and save the results in a format which is 
accessible for plotting.

You must save a 'settings.py' file in this directory before executing this pipeline.  The file needs to be 
in the format of settings_template.py.  You may copy and rename that file and make the necessary settings 
changes.  This file also determines where files will be saved and the file namings for this decoding run.  
Note that the date is important for file naming, and if you change the date or any other settings related 
to the variable 'SETTINGS_FORMAT_NAME', then the decoding is treated as a different run. 

For an example of how to perform decoding for a single session, see the file 
`decoding_example_script.py` in this directory. The instructions below detail how to run a
large-scale decoding analysis across the entire brainwide map. 

## Caching the data

We do not want to load data from ONE remotely, so we first cache all the data we need.  Subsequent pipeline calls 
to ONE can then be done in the 'local' mode.

Run the 00_data_caching.py file to do this.  Caching takes many hours, and so should be done in parallel.  We
submit 50 jobs (see N_PARA and para_index in script) using 

```
sbatch slurm_data_caching.py
```

## Caching imposter sessions

Imposter sessions are used to provide a set of pseudo regression targets.  These pseudo regression targets are
decoded to produce a null distribution for our decoding test statistics.  This step caches these targets so
that they can be accessed locally as in the previous step.  The imposter sessions may be taken from ephys 
choice world or biased choice world.  This is determined by the settings.py file parameter, i.e. whether  
'imposter_generate_from_ephys' is True of False, respectively. 
The 01_imposter_caching.py script does this and is run in parallel (default of 500 jobs) like the last step 
with the command:

```
sbatch slurm_imposter_caching.sh
```

## Generate imposter dataframe

The imposter sessions from the last step need to be aggregated into a pandas dataframe to be easily accessed 
by the rest of the pipeline.  This is done by 02_generate_imposter_df.py.  This is not done is parallel, but 
nevertheless, can be summited to a slurm cluster using

```
sbatch slurm_generate_imposter_df.py
```

## Decoding sessions

The regression is performed in this step by running 
```
03_decode_single_session.py X
```
where X is an integer which indexes across jobs.  Each sessions gets at least 1 job, so there are many 
hundreds of jobs.  These jobs are submitted in parallel using 

```
sbatch slurm_decode.sh
```

NB: some slurm clusters have a maximum number of jobs, which means you will have to submit multiple 
times.  For example, if you cannot sumbit more than 1000 jobs but require 1100, run
```
sbatch slurm_decode.sh
```
where the script contains the line 
```
#SBATCH --array=1-1000
```
Then, when you are allowed to submit more jobs run the same but change the line to
```
#SBATCH --array=1001-1100
```
.  The total number of jobs required will be the number of sessions * ceil(n_pseudo / n_pseudo_per_job).

The prefix of the filenames of output and error files found in slurm_decode.sh (lines '#SBATCH --output' 
and '#SBATCH --error') is used for post-processing.  For example, if your error filename in 
slurm_decode.sh, excluding directory, is 'ds_bwmrelease_3_.%a.err', then you will want to remember the 
prefix 'ds_bwmrelease_3_' because it is used in the next step.  Also, note the directory in which these 
files are saved and change if necessary. 

## Check decoding

Did any decoding jobs cancel (e.g. due to time limit) or did any jobs contain regression convergence 
errors? This information is in the slurm error files.  Running 
```
python 03b_check_convergence.py job_name
```
, where job_name is the error file prefix used in the previous step, will read all of these files and list 
the cancellations or convergence error warnings in the command line.  Error files are found by matching the
regex 'job_name+".*err"'.  If there are cancellations, read the corresponding error/output files and see 
why.  If it is due to time, simply repeat the previous step -- completed session/regions will be skipped 
and time will be used on the remaining session/regions.  This is typically necessary for wheel-speed 
decoding on my cluster which has a max time limit of 48 hours.

There is not a slurm file to submit this job because it is most useful to run on your current node, and the
outputs are printed directly in the REPL.

## Format results and save in dataframes

Dealing with the files saved for individual decoding runs can be unweildy.  Here, we read all the decoding 
results and aggregate the most important information into a pandas dataframe using '04_format_results.py'.  
This can take awhile so it is parallelized and submitted using
```
sbatch slurm_format.sh
```
, where the default is to parallelize 50 ways (N_PARA=50).  In that case 50 dataframes will be saved, and 
they are combined in the next step

## Generate summary tables

The final step of aggregating the data is to combine the dataframes from the previous step (e.g. 50) into a 
single dataframe which is useful for plotting.  The type of table saved depends on the variable being 
decoded (see settings.py).
