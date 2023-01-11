This code produces panels related to motor correlates of blcok of the brain-wide-map paper. It contains one python script - (fig_motor_correlates_prior.py) - and a npy file (15f742e1-1043-45c9-9504-f1e8a53c1744_left.npy) that contains an example frame being used for illustration in one panel. Please copy this file into the folder that is created when you compute the results in your one.cache_dir named brain_wide_map/motor_correlates, prior to plotting the paw position on the example frame.  

Cutting 7 behaviors during the inter-trial interval for all BWM sessions takes about 1 h. One can use one core per lag variable and compute in parallel. The lag variable specifies the start of the inter-trial interval and we compute it for two lags.

See at the bottom of the script:

```python
#    #  cut seven behaviors for all BWM sessions (used in bar plot below)
#    for lag in [-0.4, -0.6]:
#        get_PSTHs_7behaviors(lag = lag)
  

#plot bar plot summarising fraction of sessions that 
#have sig motor correlates
Result_7behave()
 
# Illustrate paw position on video frame   
paw_position_onframe()

# illustrate paw behavior per trial 
eid =  "15f742e1-1043-45c9-9504-f1e8a53c1744"
PSTH_pseudo(eid,pawex=True)  
``` 
