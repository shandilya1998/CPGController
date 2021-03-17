from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=w, n=3)}
model_dict2 = {y: fourier_series(x, f=w, n=15)}

xdata = np.linspace(-np.pi, np.pi)
ydata = np.zeros_like(xdata)
ydata[xdata > 0] = 1
# Define a Fit object for this model and data
fit = Fit(model_dict, x=xdata, y=ydata)
fit2 = Fit(model_dict2, x=xdata, y=ydata)
fit_result = fit.execute()
fit_result2 =  fit2.execute()

# Plot the result
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(xdata, ydata,color='red', label='Square Wave')
ax.plot(xdata, fit.model(x=xdata, **fit_result.params).y, color='green', ls=':', label='Fourier Reconstruction n = {n}'.format(n=3))
ax.plot(xdata, fit2.model(x=xdata, **fit_result2.params).y, color='blue', ls='--', label='Fourier Reconstruction n = {n}'.format(n=15))
ax.legend(loc = 'upper left')
fig.savefig('fourier.png')
plt.show()
