"""
Utilities for taking time series with uneven sampling or irregular time stamps and resampling
down to a fixed sample rate. This is primarily used in the context of the encoding models as a
means of adjusting the sampling of the wheel data to match the binning of the neural data.
"""

# Third party libraries
import numpy as np
from scipy import interpolate


class TimeSeries(dict):
    """A subclass of dict with dot syntax, enforcement of time stamping"""

    def __init__(self, times, values, columns=None, *args, **kwargs):
        """TimeSeries objects are explicity for storing time series data in which entry (row) has
        a time stamp associated. TS objects have obligatory 'times' and 'values' entries which
        must be passed at construction, the length of both of which must match. TimeSeries takes an
        optional 'columns' argument, which defaults to None, that is a set of labels for the
        columns in 'values'. These are also exposed via the dot syntax as pointers to the specific
        columns which they reference.

        :param times: an ordered object containing a list of timestamps for the time series data
        :param values: an ordered object containing the associated measurements for each time stamp
        :param columns: a tuple or list of column labels, defaults to none. Each column name will
            be exposed as ts.colname in the TimeSeries object unless colnames are not strings.

        Also can take any additional kwargs beyond times, values, and columns for additional data
        storage like session date, experimenter notes, etc.

        Example:
        timestamps, mousepos = load_my_data()  # in which mouspos is T x 2 array of x,y coordinates
        positions = TimeSeries(times=timestamps, values=mousepos, columns=('x', 'y'),
                               analyst='John Cleese', petshop=True,
                               notes=("Look, matey, I know a dead mouse when I see one, "
                                      'and I'm looking at one right now."))
        """
        super(TimeSeries, self).__init__(
            times=np.array(times), values=np.array(values), columns=columns, *args, **kwargs
        )
        self.__dict__ = self
        self.columns = columns
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

        # Enforce times dict key which contains a list or array of timestamps
        if len(self.times) != len(values):
            raise ValueError("Time and values must be of the same length")

        # If column labels are passed ensure same number of labels as columns, then expose
        # each column label using the dot syntax of a Bunch
        if isinstance(self.values, np.ndarray) and columns is not None:
            if self.values.shape[1] != len(columns):
                raise ValueError("Number of column labels must equal number of columns in values")
            self.update({col: self.values[:, i] for i, col in enumerate(columns)})

    def copy(self):
        """Return a new TimeSeries instance which is a copy of the current TimeSeries instance."""
        return TimeSeries(super(TimeSeries, self).copy())


def sync(
    dt, timeseries=None, offsets=None, interp="zero", fillval=np.nan
):
    """
    Function for resampling a single or multiple time series to a single, evenly-spaced, delta t
    between observations. Uses interpolation to find values.

    Can be used on brainbox.TimeSeries objects passed to the 'timeseries' kwarg.

    Uses scipy's interpolation library to perform interpolation.
    See scipy.interp1d for more information regarding interp and fillval parameters.

    :param dt: Separation of points which the output timeseries will be sampled at
    :type dt: float
    :param timeseries: A group of time series to perform alignment or a single time series.
        Must have time stamps.
    :type timeseries: tuple of TimeSeries objects, or a single TimeSeries object.
    :param offsets: tuple of offsets for time stamps of each time series. Offsets for passed
        TimeSeries objects first. Used in cases where there is a known offset between different
        sampled timeseries that must be accounted for.
        defaults to None
    :type offsets: tuple of floats, optional
    :param interp: Type of interpolation to use. Refer to scipy.interpolate.interp1d for possible
        values, defaults to 0
    :type interp: str
    :param fillval: Fill values to use when interpolating outside of range of data. See interp1d
        for possible values, defaults to np.nan
    :return: TimeSeries object with each row representing synchronized values of all
        input TimeSeries. Will carry column names from input time series if all of them have column
        names. If an array is passed no column names will be available.
    """
    #########################################
    # Checks on inputs and input processing #
    #########################################

    # Initialize a list to contain times/values pairs if no TS objs are passed
    if timeseries is None:
        timeseries = []
    # If a single time series is passed for resampling, wrap it in an iterable
    elif isinstance(timeseries, TimeSeries):
        timeseries = [timeseries]
    # Yell at the user if they try to pass stuff to timeseries that isn't a TimeSeries object
    elif not all([isinstance(ts, TimeSeries) for ts in timeseries]):
        raise TypeError(
            "All elements of 'timeseries' argument must be brainbox.TimeSeries "
            "objects. Please uses 'times' and 'values' for np.ndarray args."
        )


    # Adjust each timeseries by the associated offset if necessary then load into a list
    if offsets is not None:
        if len(offsets) != len(timeseries):
            raise ValueError("Number of offsets must equal number of timeseries")
        tstamps = [ts.times + os for ts, os in zip(timeseries, offsets)]
    else:
        tstamps = [ts.times for ts in timeseries]
    # If all input timeseries have column names, put them together for the output TS. Otherwise
    # raise an error.
    if all([ts.columns is not None for ts in timeseries]):
        colnames = []
        for ts in timeseries:
            colnames.extend(ts.columns)
    else:
        AttributeError("All input timeseries must have column names to sync")

    #################
    # Main function #
    #################

    # Get the min and max values for all timeseries combined after offsetting
    tbounds = np.array([(np.amin(ts), np.amax(ts)) for ts in tstamps])
    if not np.all(np.isfinite(tbounds)):
        # If there is a np.inf or np.nan in the time stamps for any of the timeseries this will
        # break any further code so we check for all finite values and throw an informative error.
        raise ValueError(
            "NaN or inf encountered in passed timeseries.\
                          Please either drop or fill these values."
        )
    tmin, tmax = np.amin(tbounds[:, 0]), np.amax(tbounds[:, 1])
    if fillval == "extrapolate":
        # If extrapolation is enabled we can ensure we have a full coverage of the data by
        # extending the t max to be an whole integer multiple of dt above tmin.
        # The 0.01% fudge factor is to account for floating point arithmetic errors.
        newt = np.arange(tmin, tmax + 1.0001 * (dt - (tmax - tmin) % dt), dt)
    else:
        newt = np.arange(tmin, tmax, dt)
    tsinterps = [
        interpolate.interp1d(ts.times, ts.values, kind=interp, fill_value=fillval, axis=0)
        for ts in timeseries
    ]
    syncd = TimeSeries(newt, np.hstack([tsi(newt) for tsi in tsinterps]), columns=colnames)
    return syncd
