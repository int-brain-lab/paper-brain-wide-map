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
across trials and targets are task or behavioral variables across trials.  Neurons within a given session
and region are used for a single regression.  The results from this given session and region are compared
to a non-parametric null distribution, constructed from many regressions with the same regressors and 
different targets (drawn from a null model), to determine significance.  
In order to analyze all of the BWM regions and sessions, we parallelize the 
code on a slurm based cluster.  The following pipeline steps load the data, perform the regression (both for 
the real data and to construct a non-parametric null distribution), and save the results in a format which is 
accessible for plotting.

You must save a `settings.py` file in this directory before executing this pipeline.  The said file needs to be 
in the format of `settings_template.py`.  You may copy and rename that file, making any desired settings 
changes.  The settings used for the BWM pre-print main figures can be found in the directory `settings_for_BWM_figure`. 
The settings file also determines where results will be saved and the file namings for this decoding run.  
`SETTINGS_FORMAT_NAME` plays a central role in identifying a particular decoding run.  If this variable does 
not change and the decoding pipeline is run multiple times, decoding computation which has already been 
completed will not be re-computed i.e. previous computations will be skipped.  But if this variable is 
changed then all decoding will be re-run with the pipeline.  
Notably, the `SETTINGS_FORMAT_NAME` variable can be changed simply by changing the date.  
*It is recommended, therefore, that you change the date for a fresh decoding run of a given target variable.*

Additionally, you will often be submitting "\*.sh" files to slurm, discussed below.  The slurm settings within these files may
need to be adapted for your cluster.  These "\*.sh" files contain file names and directories for the output ("\*.out") and error ("\*.err")
files saved with each job.  See the lines that begin with 
```
#SBATCH --output=
```
or
```
#SBATCH --error=
```
.  These file names within the "*.sh" files can be changed to whatever you desire (besides the .out and .err endings), but the names
should be the same.  The directory, however, for both output and error files in *each* of the "\*.sh" files
must match the `SLURM_DIR` varible found in `settings.py`.

For an example of how to perform decoding for a single session, see the file 
`decoding_example_script.py` in this directory. The `settings.py` file does not need to be copied 
or updated to run this example.

The instructions below detail how to run a large-scale decoding analysis across the entire 
brainwide map. 

## Caching the data

We do not want to load data from ONE every time we run decoding analysis.  In particular, we are concerned 
about remotely connecting to ONE with many parralel jobs such that we overload the system.  To mitigate any 
issues, we first cache all the data we need.  Subsequent pipeline calls to ONE are then done in the 
`local` mode.

The `00_data_caching.py` file is run to perform caching, but doing so in a single job takes many hours.
Instead, submit many jobs to the slurm cluster to be performed in parallel using
```
sbatch 00_slurm_data_caching.sh
``` 
Be careful not to do this with too many jobs so as to not overload the server.  We submit 50 jobs 
(see `N_PARA` and `para_index` in `00_data_caching.py`) which is not too many. 
The REPL output from with submission will print "Submitted batch job SUBMISSIONID" where SUBMISSIONID 
is a number (typically 8 digits).

Issues may arise when caching data, for example parallel jobs may not be thread safe or ONE may fail.  
Therefore, please check that data has been cached properly by running 
```
python 00b_check_download.py decodingdatacaching.SUBMISSIONID
```
where SUBMISSIONID is the number printed as described above.
If there were any issues the script will print them out.  Synchrony issues should be resolved by 
repeating the above submission another time.

## Generate session dataframe

The cached sessions from the last step need to be aggregated into a pandas dataframe so that subsequent 
steps of the decoding pipeline do not need to re-compute the bwm session list.  Computing this has led to issues
in the past because the cache is accessed with many different workers at once.

Additionally, there are commented out lines of code in this script that were used to save all the BiasedChoiceWorld 
session eids into a dataframe.  This used to be useful for the next step, but is not currently used.

The pandas dataframe is generated by the `01_generate_session_df.py` script which is submitted and run by the command, 
```
sbatch 01_slurm_generate_session_df.sh
```

## Caching imposter sessions

Imposter sessions are used to provide a set of pseudo regression targets. These pseudo regression targets are
decoded to produce a null distribution for our decoding test statistics. This step caches these targets so
that they can be accessed locally as in the "00_\*" scripts. The imposter sessions may be taken from ephys 
choice world or biased choice world. This is determined by the `settings.py` file parameter, i.e. whether  
`imposter_generate_from_ephys` is True of False, respectively. 
The `02_imposter_caching.py` script does this and is run in parallel (default of 500 jobs) with the command,

```
sbatch 02_slurm_imposter_caching.sh
```
.

## Generate imposter dataframe

The imposter sessions from the last step need to be aggregated into a pandas dataframe to be easily accessed 
by the rest of the pipeline. This is done by `03_generate_imposter_df.py`. This is not done is parallel, but 
for symmetry with the other steps of the pipeline, can be summited to a slurm cluster using

```
sbatch 03_slurm_generate_imposter_df.py
```
.

## Decoding sessions

Regression could be performed in this step through 
```
04_decode_single_session.py X
```
where `X` is an integer which indexes across jobs.  Each session gets at least 1 job, so there are many 
hundreds of jobs.  These jobs should not be run 1-by-1, but are instead submitted in parallel using 

```
sbatch 04_slurm_decode.sh
```
. 
The `04_decode_single_session.py` script is meant to be run on the cluster and not alone.  Thus, it is not 
recommended that you manipulate or control the `04_decode_single_session.py` input integers, `X`, which 
determine the session and subset of null sessions.  If you want to run only a subset of the BWM datset, 
you can filter for a subset of subjects in the BWM dataset as discussed in the next section.  
Additionally, to see an example of running a single session, see `decoding_example_script.py`.

When you submit `04_slurm_decode.sh` you should change the line 
```
#SBATCH --array=1-1000
```
where 1 is an example start index and 1000 is an example end index.
Generally you should choose the start index to be 1 and the end index to be 
`number of session \* ceil(n_pseudo / n_pseudo_per_job)`.
Some slurm clusters have a maximum number of jobs, which means the end index may be too large.
You might have to submit multiple times with different start and end indicies.  
For example, if you cannot sumbit more than 1000 jobs but require 1100 (from the calculation above), run
```
sbatch 04_slurm_decode.sh
```
where the script contains the line 
```
#SBATCH --array=1-1000
```
Then, when you are allowed to submit more jobs, run the same but change the line to
```
#SBATCH --array=1001-1100
```
. In this way, you run all the job indicies with different submissions.

The filenames of the output and error files found in `04_slurm_decode.sh` (lines `#SBATCH --output` 
and `#SBATCH --error`) are important for post-processing.  The directory needs to match `SLURM_DIR` from the 
settings file, but the filename can be chosen to have a prefix, then a period, then a number corresponding the 
the job index.  For example, if your error filename in 
`04_slurm_decode.sh` (excluding directory) is `ds_bwmrelease_3_.%a.err`, then the prefix is `ds_bwmrelease_3_`
and the job index is added by slurm through the `\%a` symbol.

You may also want to change the time limit of each job by changing the line
```
#SBATCH --time=12:00:00
```
. Typically jobs should be 12 hours (as shown), but wheel-speed and wheel-velocity take longer and should be 48 hours.
You may need to increase this time if you jobs do not finish on your cluster, or you may need to decrease the
time if your cluster has a maximum time limit.

## Check decoding

Did any decoding jobs cancel (e.g. due to time limit) or did any jobs contain regression convergence 
errors? This information is in the slurm error files.  Run 
```
python 04b_check_convergence.py job_name_prefix
```
, where `job_name_prefix` is the filename prefix from the previous step.  This will read all 
output and error files and print a list of  
the cancellations or convergence error warnings in the REPL.  Error files are found by matching the
regex `job_name+".*err"`.  If there are cancellations, read the corresponding error/output files and see 
why.  If it is due to time, simply repeat the previous step -- completed session/regions will be skipped 
and time will be used on the remaining session/regions.  This is typically necessary for wheel-speed which
often takes more than 48 hours to complete despite the fact that my (Brandon Benson's) cluster (Stanford Sherlock)
has a max time limit of 48 hours.

## Format results and save in dataframes

Dealing with the files saved for individual decoding runs can be unweildy.  Here, we read all the decoding 
results and aggregate the most important information into a pandas dataframe using `05_format_results.py`.  
This can take awhile so it is parallelized and submitted using
```
sbatch 05_slurm_format.sh
```
where the default is to parallelize 50 ways (`N_PARA=50`).  In that case 50 dataframes will be saved, and 
they are combined in the next step.  
When the above command is run, there will be a printed output, "Submitted batch job FORMATSUBMISSIONID"
where FORMATSUBMISSIONID is a number that corresponds to the submission.

Check that this step has completed properly using the following line of code
```
python 05b_check_formats.py decodingformat.FORMATSUBMISSIONID
```
where FORMATSUBMISSIONID is the number explained above. It is assumed that the output and error file prefixes
fount in `05_slurm_format.sh` are both `decodingformat`.  If you change this name, you will have to use that 
file prefix to replace `decodingformat` in the above line of code.  
Running that in the REPL will output a list of non-successful files.
Typically this list is empty, so you can proceed to the next step.  If it is not, you should investigate the
listed output and error files.

## Generate summary tables

The final step of aggregating the data is to combine the many dataframes from the previous step (e.g. 50 dataframes) into a 
single dataframe which is useful for plotting.  The type of table saved depends on the variable being 
decoded (see `settings.py`).  Again, this can be accomplished by running
```
sbatch 06_slurm_generate_summary.sh
```
, which runs the file `06_generate_summary_tables.py` on the cluster.
Please consider the RAM that you use for this step.  It is typically enough to use 16GB, which is achieved by the line in 
the "\*.sh" file that reads
```
#SBATCH --mem=16G
```
. The wheel-speed and wheel-velocity summary tables, however, take more than 64GB of RAM to construct, 
so set the line in `06_slurm_generate_summary.sh` to read
```
#SBATCH --mem=128G
```
. *Hopefully your cluster allows you to use this much RAM, or else you will have to find a different way to 
achieve this step*

## Run a subset of the BWM dataset

Running a subset of the BMW dataset can be helpful for debugging or quick analysis purposes.
You can filter for a subset of subjects in the BWM dataset by adjusting `04_slurm_decode.sh`.
Specifically, comment out this line
```
python 04_decode_single_session.py $SLURM_ARRAY_TASK_ID
```
and uncomment the last line of the file.  This adds a series of subjects as inputs to
the `04_decode_single_session.py` script, and you can add or remove any number of these subjects.  The script
will filter for only BWM sessions of these subjects.

To see where the filtering occurs in `04_decode*.py` look for the if statement `if len(sys.argv) > 2:` 
and the comment above it.

Additionally, to see an example of running a single sessions, see `decoding_example_script.py`.


## Omnibus decoding

Omnibus decoding refers to decoding that is done to draw a statistical conclusion about whether or not a variable
can be decoded from the entire BWM dataset (rather than within a single region/session).  In order to do such decoding, 
the typical pipeline must be changed to be agnostic to regions.  Since regions are ignored, the regressors
should be all the neurons within a given eid i.e. neurons should be combined across probes and regions into a single omnibus
region.

In order to effect these changes, the settings file line that contains SINGLE_REGION must be changed to 
```
SINGLE_REGION = False #...
```
There should also be a line of code in the settings file that contains CANONICAL_SET which should be set to
```
CANONICAL_SET = True
```
to ensure that only neurons from the canonical set are used for omnibus decoding.

The settings used for the omnibus test in the BWM pre-print are included in the folder "settings_for_BWM_omnibus".
