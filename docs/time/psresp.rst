*****************************
Power spectral response method
*****************************
.. currentmodule:: gammapy.time.psresp

Introduction
============
`~gammapy.time.psresp` establishes the power spectral response method (PSRESP) for the analysis of the power spectral density (PSD) of light curves.
It is a Monte Carlo approach that takes the sampling directly into account and reveals the underlying PSD.
As model for the PSD, an unbroken power law is assumed.
For each model parameter (i.e. slope of the power law), a success fraction (SUF) is calculated that defines the inverse rejection level for this model.
The slope of the PSD model for the light curve is estimated as the mean of all slopes providing a significant SUF.
The corresponding error is given by the full width at half maximum (FWHM) of the SUF distribution.

Getting Started
===============
Input
-----
`~gammapy.time.psresp` takes a light curve in format time, flux and flux error.
For the PSD model, the trial slopes have to be forwarded via `slopes`.
The PSRESP method bins the light curve and the periodogram as defined by `dt` and `df`.
To determine the significant SUF, the percentile for the SUF distribution, `percentile`, needs to be given.
The number of simulations can be defined by `number_simulations`, it is 100 by default.
Additionally, the oversampling of the artificial light curves can be defined by `oversampling`.
`~gammapy.time.plot_psresp` takes the output of `~gammapy.time.psresp` as input.

Output
------
`~gammapy.time.psres` returns the mean slope and its error,
the success fraction over a grid of model parameters (`slopes`, `dt`, `df`),
parameters `dt` and `df` providing a significant SUF
and the statistics used to calculate the mean slope and its error.

Example
=======
An example of detecting a period is shown in the figure below. The light curve is from the X-ray binary LS 5039 observed with H.E.S.S. at energies above 0.1 TeV in 2005 [1]_. The Lomb-Scargle reveals the period of :math:`(3.907 \pm 0.001)` days in agreement with [1]_ and [2]_.

.. gp-extra-image:: lomb_scargle_long.png
   :width: 100 %
   :alt: alternate text
   :align: left

The periodogram shows fluctuations for small periods and a smoothed behaviour for longer periods that are due to sampling effects and aliasing.
If this is the case, `max_period` can be defined to limit the period range for the analysis.
This way, the resoultion can be increased with equal computation time.

.. gp-extra-image:: lomb_scargle_short.png
   :width: 100 %
   :alt: alternate text
   :align: left

The periodogram has many spurious peaks, which are due to several factors:

1. Errors in observations lead to leakage of power from the true peaks.
2. The signal is not a perfect sinusoid, so additional peaks can indicate higher-frequency components in the signal.
3. The spectral window function shows two prominent peaks around one and 27 days. The first one arises from the nightly observation cycle, the second from the lunar phase. Thus, aliases are expected to appear at :math:`f_{{alias}} = f_{{true}} + n f_{{window}}` for integer values of :math:`n`. For the peak in the spectral window function at :math:`f_{{window}} = 1 day^{{-1}}`, this corresponds to the third highest peak in the periodogram at :math:`p_{{alias}} = 0.796`.

The returned significance must be used with caution. If the resolution is too rough, several periods will be detected with a significance of 100 per cent. Thus, an eyesight inspection is obligatory.

.. [1] F. Aharonian, 3.9 day orbital modulation in the TeV gamma-ray flux and spectrum from the X-ray binary LS 5039,
   `Link <https://www.aanda.org/articles/aa/pdf/forth/aa5940-06.pdf>`_
.. [2] J. Casares, A possible black hole in the gamma-ray microquasar LS 5039,
   `Link <https://academic.oup.com/mnras/article/364/3/899/1187228/A-possible-black-hole-in-the-ray-microquasar-LS>`_
