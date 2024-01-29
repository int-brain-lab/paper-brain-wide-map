from brainwidemap.manifold import state_space_bwm


for split in ['choice', 'stim', 'fback', 'block']:
    # computes PETHs, distance sums
    state_space_bwm.get_all_d_vars(split)
    # combine results across insertions
    state_space_bwm.d_var_stacked(split)

# this replicates the main figure of the paper
state_space_bwm.plot_all()
