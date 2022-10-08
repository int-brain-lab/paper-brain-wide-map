# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

# IBL libraries
import brainbox.io.one as bbone
from neurencoding.utils import full_rcos
from one.api import ONE

colors = {
    'Red': '#e6194B',
    'Green': '#3cb44b',
    'Yellow': '#ffe119',
    'Blue': '#4363d8',
    'Orange': '#f58231',
    'Cyan': '#42d4f4',
    'Magenta': '#f032e6',
    'Pink': '#fabed4',
    'Teal': '#469990',
    'Lavender': '#dcbeff',
    'Brown': '#9A6324',
    'Beige': '#fffac8',
    'Maroon': '#800000',
    'Mint': '#aaffc3',
    'Navy': '#000075',
    'Grey': '#a9a9a9',
    'White': '#ffffff',
    'Black': '#000000'
}

one = ONE()
eid = "297bd519-78f8-45d2-af85-835e865e228f"
trialsdf = bbone.load_trials_df(eid,
                                t_before=0.6,
                                t_after=0.6,
                                addtl_types=['firstMovement_times'],
                                ret_abswheel=True)
extrial = trialsdf.loc[113]
bases = full_rcos(0.4, 5, binf := lambda t: np.ceil(t / .02).astype(int))
wheelbases = full_rcos(0.3, 3, binf)
fmovebases = full_rcos(0.2, 3, binf)

stimw = np.array([-0.2, 0, 0.2, 0.8, 0.2])
fdbkw = np.array([0, 0.2, 0.5, 0.5, 0.2])
whlw = np.array([-0.5, 0, 0.3])
fmovew = np.array([0.9, 0, 0])
stimk = bases @ stimw
fdbkk = bases @ fdbkw
fmovew = fmovebases @ fmovew
wheelk = wheelbases @ whlw

wheeltrace = extrial.wheel_velocity
wheelcomponent = np.convolve(wheelk, wheeltrace, mode='same')

iti = ((extrial.stimOn_times - 0.4) - extrial.trial_start, (extrial.stimOn_times - 0.1) - extrial.trial_start)
tr = ((extrial.stimOn_times - extrial.trial_start), (extrial.trial_end - extrial.trial_start))
ititrace = np.zeros_like(wheeltrace)
ititrace[binf(iti[0]):binf(iti[1])] = 0.2
trtrace = np.zeros_like(wheeltrace)
trtrace[binf(tr[0]): binf(extrial.feedback_times - extrial.trial_start)] = 0.2
trtrace[binf(extrial.feedback_times - extrial.trial_start): binf(tr[1])] = 0.8

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
ax.vlines(s := extrial.stimOn_times, 0, 1, color=colors['Maroon'], lw=2.5)
# ax.text(s='Stimulus onset', x=s, y=-0.45, rotation=-45, fontsize=15, color=colors["Maroon"])
ax.vlines(f := extrial.feedback_times, 0, 1, color=colors['Teal'], lw=2.5)
# ax.text(s='Feedback time', x=f, y=-0.45, rotation=-45, fontsize=15, color=colors["Teal"])
ax.vlines(fm := extrial.firstMovement_times, 0, 1, color=colors['Navy'], lw=2.5)
# ax.text(s='Movement onset', x=fm, y=-0.45, rotation=-45, fontsize=15, color=colors["Navy"])
ax.plot(np.arange(s, s + 0.4, 0.02),
        stimk,
        color=colors['Maroon'],
        ls='--',
        lw=2.5,
        label='Stimulus kernel')
ax.plot(np.arange(f, f + 0.4, 0.02),
        fdbkk,
        color=colors['Teal'],
        ls="--",
        lw=2.5,
        label='Feedback kernel')
ax.plot(np.arange(fm - 0.2, fm - 1e-6, 0.02),
        fmovew,
        color=colors['Navy'],
        ls="--",
        lw=2.5,
        label='First movement kernel')
ax.plot(t := np.arange(extrial.trial_start, extrial.trial_end, 0.02),
        wheelcomponent * 2 - 0.2,
        ls="--",
        lw=2.5,
        label='Wheel component')
ax.plot(t, ititrace, ls="--", lw=2.5, label='ITI P(Left)')
ax.plot(t, trtrace, ls="--", lw=2.5, label='Trial P(Left)')
ax.hlines(0, extrial.trial_start, extrial.trial_end, color=colors['Black'], lw=2.5)

ax = fig.add_subplot()
sumkerns = 
