This code produces panels related to motor correlates of blcok of the brain-wide-map paper. It contains one python script - (fig\_motor\_correlates\_prior.py) - and a npy file (15f742e1-1043-45c9-9504-f1e8a53c1744_left.npy) that contains an example frame being used for illustration in one panel.

Cutting 7 behaviors during the inter-trial interval for all BWM sessions takes about 1 h. One can use one core per lag variable and compute in parallel. The lag variable specifies the start of the inter-trial interval and we compute it for two lags.

See at the bottom of the script:
Computing the behaviors for all sessions, 
needed for the summary bar plot.

```python
#    #  cut seven behaviors for all BWM sessions
#    for lag in [-0.4, -0.6]:
#        get_PSTHs_7behaviors(lag = lag)
```   

#plot bar plot summarising fraction of sessions that 
#have sig motor correlates
Result_7behave()
 
# Illustrate paw position on video frame   
paw_position_onframe()

# illustrate paw behavior per trial 
eid =  "15f742e1-1043-45c9-9504-f1e8a53c1744"
PSTH_pseudo(eid,pawex=True)  
