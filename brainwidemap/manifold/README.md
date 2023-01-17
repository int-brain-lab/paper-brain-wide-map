The script state_space_bwm.py is used to compute PETHs and subsequent manifold analysis. 
Data processing and plotting both are implemented by this single script.

To compute the results, first PETHs are saved for all BWM sessions in a subfolder of the local flatiron folder.
This can be done for different task variables in parallel, or in sequence as by this command in an ipython session:

```python:
run state_space_bwm.py
for split in align:  # you can run these splits on different cores, then it's 3h 
    get_all_d_vars(split)
    d_var_stacked(split)
```

The results are save din an extra results folder on the local flatiron folder. They can then be plotted by running:

```python:
plot_all()
```
