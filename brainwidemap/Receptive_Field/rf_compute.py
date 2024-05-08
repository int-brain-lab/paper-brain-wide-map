'''
Compute the significance and parameters of a visual receptive field
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import leastsq


def _gaussian_function_2d(peak_height, center_y, center_x, width_y, width_x):
    """Returns a 2D Gaussian function

    Parameters
    ----------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis

    Returns
    -------
    f(x,y) : function
        Returns the value of the distribution at a
        particular x,y coordinate

    """

    return lambda y, x: peak_height * np.exp(
        -(((center_y - y) / width_y) ** 2 + ((center_x - x) / width_x) ** 2)
        / 2
    )


def rf_guassian(dim_y, dim_x, peak_height, center_y, center_x, width_y, width_x):
    """Returns a RF matrix based on 2D Gaussian function

    Parameters
    ----------
    dim_y :
        length of y-position
    dim_x :
        length of x-position

    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis

    Returns
    -------
    rf_matrix : RF matrix

    """
    f = _gaussian_function_2d(peak_height, center_y, center_x, width_y, width_x)
    rf_matrix = np.zeros((dim_y, dim_x))
    for i in range(dim_y):
        for j in range(dim_x):
            rf_matrix[i, j] = f(y=i, x=j)

    return rf_matrix


def gaussian_moments_2d(data):
    """Finds the moments of a 2D Gaussian distribution,
    given an input matrix

    Parameters
    ----------
    data : numpy.ndarray
        2D matrix

    Returns
    -------
    baseline:
        baseline of data
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    """

    total = data.sum()
    height = data.max()

    Y, X = np.indices(data.shape)
    center_y = (Y * data).sum() / total
    center_x = (X * data).sum() / total

    if (
            np.isnan(center_y)
            or np.isinf(center_y)
            or np.isnan(center_x)
            or np.isinf(center_x)
    ):
        return None

    col = data[:, int(center_x)]
    row = data[int(center_y), :]

    width_y = np.sqrt(
        np.abs((np.arange(row.size) - center_y) ** 2 * row).sum() / row.sum()
    )
    width_x = np.sqrt(
        np.abs((np.arange(col.size) - center_x) ** 2 * col).sum() / col.sum()
    )

    return height, center_y, center_x, width_y, width_x


def fit_2d_gaussian(matrix):
    """Fits a receptive field with a 2-dimensional Gaussian
    distribution

    Parameters
    ----------
    matrix : numpy.ndarray
        2D matrix of spike counts

    Returns
    -------
    parameters - tuple
        peak_height : peak of distribution
        center_y : y-coordinate of distribution center
        center_x : x-coordinate of distribution center
        width_y : width of distribution along x-axis
        width_x : width of distribution along y-axis
        rsq :     r-square value of fitting
    success - bool
        True if a fit was found, False otherwise
    """

    params = gaussian_moments_2d(abs(matrix))
    if params is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan), False

    def errorfunction(p):
        return np.ravel(
            _gaussian_function_2d(*p)(*np.indices(matrix.shape)) - matrix
        )

    fit_params, ier = leastsq(errorfunction, params)
    success = True if ier < 5 else False

    RSS = (errorfunction(fit_params) ** 2).sum()
    TSS = ((matrix - np.mean(matrix)) ** 2).sum()
    rsq = 1 - RSS / TSS

    return rsq, fit_params, success


def fit_2d_gaussian_stats(matrix, nShuffle=1000):
    """Compute empirical p-value of fiting a receptive field with a 2-dimensional Gaussian
    distribution

    Parameters
    ----------
    matrix : numpy.ndarray
        2D matrix of spike counts
    nShuffle: number of shuffles

    Returns
    -------
    p_value: empirical p-value of rsq

    parameters - tuple
        peak_height : peak of distribution
        center_y : y-coordinate of distribution center
        center_x : x-coordinate of distribution center
        width_y : width of distribution along x-axis
        width_x : width of distribution along y-axis
        rsq :     r-square value of fitting
    success - bool
        True if a fit was found, False otherwise
    """

    rsq, fit_params, success = fit_2d_gaussian(matrix)

    length_1 = len(matrix[:, 0])
    length_2 = len(matrix[0, :])

    shuffle_rsq = np.zeros(nShuffle)
    for i_shuffle in range(nShuffle):
        temp_matrix_1 = matrix.reshape((length_1 * length_2, 1))
        temp_matrix_2 = np.random.permutation(temp_matrix_1)
        temp_matrix_3 = temp_matrix_2.reshape((length_1, length_2))

        temp_rsq, temp_fit_params, temp_success = fit_2d_gaussian(temp_matrix_3)
        shuffle_rsq[i_shuffle] = temp_rsq

    samples = len(np.argwhere(shuffle_rsq > rsq))
    p_value = (samples + 1) / (nShuffle)

    return p_value, rsq, fit_params


def is_rf_inverted(rf_thresh):
    """Checks if the receptive field mapping timulus is suppressing
    or exciting the cell

    Parameters
    ----------
    rf_thresh : matrix
        matrix of spike counts at each stimulus position

    Returns
    -------
    if_rf_inverted : bool
        True if the receptive field is inverted
    """
    edge_mask = np.zeros(rf_thresh.shape)

    edge_mask[:, 0] = 1
    edge_mask[:, -1] = 1
    edge_mask[0, :] = 1
    edge_mask[-1, :] = 1

    num_edge_pixels = np.sum(rf_thresh * edge_mask)

    return num_edge_pixels > np.sum(edge_mask) / 2


def invert_rf(rf):
    """Creates an inverted version of the receptive field

    Parameters
    ----------
    rf - matrix of spike counts at each stimulus position

    Returns
    -------
    rf_inverted - new RF matrix

    """
    return np.max(rf) - rf


def threshold_rf(rf, threshold):
    """Creates a spatial mask based on the receptive field peak
    and returns the x, y coordinates of the center of mass,
    as well as the area.

    Parameters
    ----------
    rf : numpy.ndarray
        2D matrix of spike counts
    threshold : float
        Threshold as ratio of the RF's standard deviation

    Returns
    -------
    threshold_rf : numpy.ndarray
        Thresholded version of the original RF
    center_x : float
        x-coordinate of mask center of mass
    center_y : float
        y-coordinate of mask center of mass
    area : float
        area of mask
    """
    rf_filt = ndi.gaussian_filter(rf, 1)

    threshold_value = np.max(rf_filt) - np.std(rf_filt) * threshold

    rf_thresh = np.zeros(rf.shape, dtype="bool")
    rf_thresh[rf_filt > threshold_value] = True

    labels, num_features = ndi.label(rf_thresh)

    best_label = np.argmax(
        ndi.maximum(rf_filt, labels=labels, index=np.unique(labels))
    )

    labels[labels != best_label] = 0
    labels[labels > 0] = 1

    center_y, center_x = ndi.measurements.center_of_mass(labels)
    area = float(np.sum(labels))

    return labels, np.around(center_x, 4), np.around(center_y, 4), area


def rf_on_screen(rf, center_y, center_x):
    """Checks whether the receptive field is on the screen, given the
    center location."""
    return 0 < center_y < rf.shape[0] and 0 < center_x < rf.shape[1]


def convert_elevation_to_degrees(
        elevation_in_pixels, elevation_offset_degrees=-30
):
    """Converts a pixel-based elevation into degrees relative to
    center of gaze

    The receptive field computed by this class is oriented such
    that the pixel values are in the correct relative location
    when using matplotlib.pyplot.imshow(), which places (0,0)
    in the upper-left corner of the figure.

    Therefore, we need to invert the elevation value prior
    to converting to degrees.

    Parameters
    ----------
    elevation_in_pixels : float
    elevation_offset_degrees: float

    Returns
    -------
    elevation_in_degrees : float
    """
    elevation_in_degrees = (
            convert_pixels_to_degrees(8 - elevation_in_pixels)
            + elevation_offset_degrees
    )

    return elevation_in_degrees


def convert_azimuth_to_degrees(azimuth_in_pixels, azimuth_offset_degrees=10):
    """Converts a pixel-based azimuth into degrees relative
    to center of gaze

    Parameters
    ----------
    azimuth_in_pixels : float
    azimuth_offset_degrees: float

    Returns
    -------
    azimuth_in_degrees : float
    """
    azimuth_in_degrees = (
            convert_pixels_to_degrees((azimuth_in_pixels)) + azimuth_offset_degrees
    )

    return azimuth_in_degrees


def convert_pixels_to_degrees(value_in_pixels, degrees_to_pixels_ratio=10):
    """Converts a pixel-based distance into degrees

    Parameters
    ----------
    value_in_pixels : float
    degrees_to_pixels_ratio: float

    Returns
    -------
    value in degrees : float
    """
    return value_in_pixels * degrees_to_pixels_ratio
