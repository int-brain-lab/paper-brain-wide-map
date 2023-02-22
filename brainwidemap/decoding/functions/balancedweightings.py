import numpy as np
import openturns
from sklearn.utils.class_weight import compute_sample_weight


def pdf_from_histogram(x, out):
    # unit test of pdf_from_histogram
    # out = np.histogram(np.array([0.9, 0.9]), bins=target_distribution[-1], density=True)
    # out[0][(np.array([0.9])[:, None] > out[1][None]).sum(axis=-1) - 1]
    return out[0][(x[:, None] > out[1][None]).sum(axis=-1) - 1]


def balanced_weighting(vec, continuous, use_openturns, target_distribution):
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.KernelSmoothing.html?highlight=kernel%20smoothing
    # This plug-in method for bandwidth estimation is based on the solve-the-equation rule from
    # (Sheather, Jones, 1991).
    if continuous:
        if use_openturns:
            factory = openturns.KernelSmoothing()
            sample = openturns.Sample(vec[:, None])
            bandwidth = factory.computePluginBandwidth(sample)
            distribution = factory.build(sample, bandwidth)
            proposal_weights = np.array(distribution.computePDF(sample)).squeeze()
            balanced_weight = np.ones(vec.size) / proposal_weights
        else:
            emp_distribution = np.histogram(vec, bins=target_distribution[-1], density=True)
            balanced_weight = pdf_from_histogram(vec, target_distribution) / pdf_from_histogram(vec, emp_distribution)
    else:
        balanced_weight = compute_sample_weight("balanced", y=vec)
    return balanced_weight
