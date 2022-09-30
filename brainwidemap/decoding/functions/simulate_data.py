import numpy as np

x_range = np.arange(len(msub_binned))
polyfit = np.polyfit(x_range, msub_binned, deg=3)
ypred = np.polyval(polyfit, x_range[:, None])
ypred = np.maximum(
    np.nan_to_num((tvec[mask][:, None] * ypred) /
                  (tvec[mask][:, None] * ypred).mean(axis=0) *
                  np.mean(msub_binned, axis=0)[None], 0), 0)
msub_binned = np.random.poisson(ypred)
'''
from matplotlib import pyplot as plt
plt.figure()
iplot=12
plt.plot(msub_binned[:, iplot]); plt.plot(ypred[:, iplot])
plt.draw(); plt.show()
'''