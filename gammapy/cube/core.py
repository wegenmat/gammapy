# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gamma-ray spectral cube: longitude, latitude and spectral axis.

TODO: split `SkyCube` into a base class ``SkyCube`` and a few sub-classes:

* ``SkyCube`` to represent functions evaluated at grid points (diffuse model format ... what is there now).
* ``ExposureCube`` should also be supported (same semantics, but different units / methods as ``SkyCube`` (``gtexpcube`` format)
* ``SkyCubeHistogram`` to represent model or actual counts in energy bands (``gtbin`` format)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.wcs import WCS
from astropy.utils import lazyproperty

from ..utils.scripts import make_path
from ..utils.testing import assert_wcs_allclose
from ..utils.energy import EnergyBounds, Energy
from ..utils.fits import table_to_fits_table
from ..image import SkyImage
from ..spectrum import LogEnergyAxis
from ..spectrum.utils import _trapz_loglog

__all__ = ['SkyCube']


class SkyCube(object):
    """Sky cube with dimensions lon, lat and energy.

    Can be used e.g. for Fermi or GALPROP diffuse model cubes.

    Note that there is a very nice ``SkyCube`` implementation here:
    http://spectral-cube.readthedocs.io/en/latest/index.html

    Here is some discussion if / how it could be used:
    https://github.com/radio-astro-tools/spectral-cube/issues/110

    For now we re-implement what we need here.

    The order of the sky cube axes can be very confusing ... this should help:

    * The ``data`` array axis order is ``(energy, lat, lon)``.
    * The ``wcs`` object is a two dimensional celestial WCS with axis order ``(lon, lat)``.

    Parameters
    ----------
    name : str
        Name of the sky cube.
    data : `~astropy.units.Quantity`
        Data array (3-dim)
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array

    Attributes
    ----------
    data : `~astropy.units.Quantity`
        Data array (3-dim)
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy values, at the center of the bin.
    energy_axis : `~gammapy.spectrum.LogEnergyAxis`
        Energy axis
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.


    Notes
    -----
    Diffuse model files in this format are distributed with the Fermi Science tools
    software and can also be downloaded at
    http://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html

    E.g. the 2-year diffuse model that was used in the 2FGL catalog production is at
    http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gal_2yearp7v6_v0.fits
    """

    def __init__(self, name=None, data=None, wcs=None, energy=None, meta=None):
        # TODO: check validity of inputs
        self.name = name
        self.data = data
        self.wcs = wcs
        self.meta = meta

        # TODO: decide whether we want to use an EnergyAxis object or just use the array directly.
        if energy:
            self.energy_axis = LogEnergyAxis(energy)

    @lazyproperty
    def _interpolate_data(self):
        """
        Interpolate data using `~scipy.interpolate.RegularGridInterpolator`)
        """
        from scipy.interpolate import RegularGridInterpolator

        # set up log interpolation
        unit = self.data.unit
        # TODO: how to deal with zero values?
        log_data = np.log(self.data.value)
        z = np.arange(self.data.shape[0])
        y = np.arange(self.data.shape[1])
        x = np.arange(self.data.shape[2])

        f = RegularGridInterpolator((z, y, x), log_data, fill_value=None,
                                    bounds_error=False)

        def interpolate(z, y, x, method='linear'):
            shape = z.shape
            coords = np.column_stack([z.flat, y.flat, x.flat])
            val = f(coords, method=method)
            return Quantity(np.exp(val).reshape(shape), unit)

        return interpolate

    @classmethod
    def read_hdu(cls, hdu_list):
        """Read sky cube from HDU.

        Parameters
        ----------
        object_hdu : `~astropy.io.fits.ImageHDU`
            Image HDU object to be read
        energy_table_hdu : `~astropy.io.fits.TableHDU`
            Table HDU object giving energies of each slice
            of the Image HDU object_hdu

        Returns
        -------
        sky_cube : `SkyCube`
            Sky cube
        """
        object_hdu = hdu_list[0]
        energy_table_hdu = hdu_list[1]
        data = object_hdu.data
        data = Quantity(data, '1 / (cm2 MeV s sr)')
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and call ENERGY from the table extension
        header = object_hdu.header
        wcs = WCS(header)
        energy = energy_table_hdu.data['Energy']
        energy = Quantity(energy, 'MeV')
        meta = OrderedDict(header)
        return cls(data=data, wcs=wcs, energy=energy, meta=meta)

    @classmethod
    def read(cls, filename, format='fermi'):
        """Read sky cube from FITS file.

        Parameters
        ----------
        filename : str
            File name
        format : {'fermi', 'fermi-counts'}
            Fits file format.

        Returns
        -------
        sky_cube : `SkyCube`
            Sky cube
        """
        filename = str(make_path(filename))
        data = fits.getdata(filename)
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and do ENERGY ourselves
        header = fits.getheader(filename)
        wcs = WCS(header).celestial
        meta = OrderedDict(header)
        if format == 'fermi-background':
            energy = Table.read(filename, 'ENERGIES')['Energy']
            energy = Quantity(energy, 'MeV')
            data = Quantity(data, '1 / (cm2 MeV s sr)')
            name = 'flux'
        elif format == 'fermi-counts':
            energy = EnergyBounds.from_ebounds(fits.open(filename)['EBOUNDS'], unit='keV')
            energy = energy.log_centers
            data = Quantity(data, 'count')
            name = 'counts'
        elif format == 'fermi-exposure':
            pass
            name = 'exposure'

        else:
            raise ValueError('Not a valid cube fits format')

        return cls(name=name, data=data, wcs=wcs, energy=energy, meta=meta)

    def fill_events(self, events, weights=None):
        """
        Fill events (modifies ``data`` attribute).

        Parameters
        ----------
        events : `~gammapy.data.EventList`
            Event list
        weights : str, optional
            Column to use as weights (none by default)
        """
        if weights is not None:
            weights = events[weights]

        xx, yy, zz = self.wcs_skycoord_to_pixel(events.radec, events.energy)

        bins = self._bins_energy, self.sky_image_ref._bins_pix[0], self.sky_image_ref._bins_pix[1]
        data = np.histogramdd([zz, yy, xx], bins, weights=weights)[0]

        self.data = self.data + data

    @property
    def _bins_energy(self):
        return np.arange(self.data.shape[0] + 1)

    @classmethod
    def empty(cls, emin=0.5, emax=100, enumbins=10, eunit='TeV', **kwargs):
        """
        Create empty sky cube with log equal energy binning from the scratch.

        Parameters
        ----------
        emin : float
            Minimum energy.
        emax : float
            Maximum energy.
        enumbins : int
            Number of energy bins.
        eunit : str
            Energy unit.
        kwargs : dict
            Keyword arguments passed to `~gammapy.image.SkyImage.empty` to create
            the spatial part of the cube.
        """
        image = SkyImage.empty(**kwargs)
        energy = EnergyBounds.equal_log_spacing(emin, emax, enumbins, eunit)
        data = image.data * np.ones(len(energy)).reshape((-1, 1, 1)) * u.Unit('')
        return cls(data=data, wcs=image.wcs, energy=energy)

    @classmethod
    def empty_like(cls, refcube, fill=0):
        """
        Create an empty sky cube with the same WCS, energy specification and meta
        as given sky cube.

        Parameters
        ----------
        refcube : `~gammapy.cube.SkyCube`
            Reference sky cube.
        fill : float, optional
            Fill image with constant value. Default is 0.
        """
        wcs = refcube.wcs.copy()
        data = fill * np.ones_like(refcube.data)
        energies = refcube.energies.copy()
        return cls(data=data, wcs=wcs, energy=energies, meta=refcube.meta)

    def wcs_skycoord_to_pixel(self, position, energy):
        """Convert world to pixel coordinates.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        (z, y, x) : tuple
            Tuple of x, y, z coordinates.
        """
        if not position.shape == energy.shape:
            raise ValueError('Position and energy array must have the same shape.')

        x, y = self.sky_image_ref.wcs_skycoord_to_pixel(position)
        z = self.energy_axis.world2pix(energy)

        #TODO: check order, so that it corresponds to data axis order
        return (x, y, z)

    def wcs_pixel_to_skycoord(self, x, y, z):
        """Convert pixel to world coordinates.

        Parameters
        ----------
        x, y, z

        Returns
        -------
        lon, lat, energy
        """
        position = self.sky_image_ref.wcs_pixel_to_skycoord(x, y)
        energy = self.energy_axis.pix2world(z)
        energy = Quantity(energy, self.energy_axis.energy.unit)
        return (position, energy)

    def to_sherpa_data3d(self):
        """
        Convert sky cube to sherpa `Data3D` object.
        """
        from .sherpa_ import Data3D

        # Energy axes
        energies = self.energy_axis.energy.to("TeV").value
        ebounds = EnergyBounds(Quantity(energies, 'TeV'))
        elo = ebounds.lower_bounds.value
        ehi = ebounds.upper_bounds.value

        coordinates = self.sky_image_ref.coordinates()
        ra = coordinates.data.lon
        dec = coordinates.data.lat

        n_ebins = len(elo)
        ra_cube = np.tile(ra, (n_ebins, 1, 1))
        dec_cube = np.tile(dec, (n_ebins, 1, 1))
        elo_cube = elo.reshape(n_ebins, 1, 1) * np.ones_like(ra)
        ehi_cube = ehi.reshape(n_ebins, 1, 1) * np.ones_like(ra)

        return Data3D('', elo_cube.ravel(), ehi_cube.ravel(), ra_cube.ravel(),
                      dec_cube.ravel(), self.data.value.ravel(),
                      self.data.value.shape)

    def sky_image(self, energy, interpolation=None):
        """
        Slice a 2-dim `~gammapy.image.SkyImage` from the cube at a given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy value
        interpolation : {None, 'linear', 'nearest'}
            Interpolate data values between energies. None corresponds to
            'nearest', but might have advantages in performance, because
            no interpolator is set up.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            2-dim sky image
        """
        # TODO: should we pass something in SkyImage (we speak about meta)?
        z = self.energy_axis.world2pix(energy)

        if interpolation:
            y = np.arange(self.data.shape[1])
            x = np.arange(self.data.shape[2])
            z, y, x = np.meshgrid(z, y, x, indexing='ij')
            data = self._interpolate_data(z, y, x)[0]
        else:
            data = self.data[int(z)]
        return SkyImage(name=self.name, data=data, wcs=self.wcs)

    @lazyproperty
    def sky_image_ref(self):
        """
        Empty reference `~gammapy.image.SkyImage`.

        Examples
        --------
        Can be used to access the spatial information of the cube:

            >>> from gammapy.cube import SkyCube
            >>> cube = SkyCube.empty()
            >>> coords = cube.ref_sky_image.coordinates()
            >>> solid_angle = cube.ref_sky_image.solid_angle()

        """
        wcs = self.wcs.celestial
        data = np.zeros_like(self.data[0])
        return SkyImage(name=self.name, data=data, wcs=wcs)

    def lookup(self, position, energy, interpolation=False):
        """Differential flux.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential flux (1 / (cm2 MeV s sr))
        """
        # TODO: add interpolation option using NDDataArray

        z, y, x = self.wcs_skycoord_to_pixel(position, energy)

        if interpolation:
            vals = self._interpolate_data(z, y, x)
        else:
            vals = self.data[np.rint(z).astype('int'), np.rint(y).astype('int'),
                             np.rint(x).astype('int')]
        return vals

    def show(self, viewer='mpl', ds9options=None, **kwargs):
        """
        Show sky cube in image viewer.

        Parameters
        ----------
        viewer : {'mpl', 'ds9'}
            Which image viewer to use. Option 'ds9' requires ds9 to be installed.
        ds9options : list, optional
            List of options passed to ds9. E.g. ['-cmap', 'heat', '-scale', 'log'].
            Any valid ds9 command line option can be passed.
            See http://ds9.si.edu/doc/ref/command.html for details.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.
        """
        import matplotlib.pyplot as plt
        from ipywidgets import interact

        if viewer == 'mpl':
            max_ = self.data.shape[0] - 1

            def show_image(idx):
                energy = self.energy_axis.pix2world(idx)
                image = self.sky_image(energy)
                image.data = image.data.value
                image.show(**kwargs)

            return interact(show_image, idx=(0, max_, 1))
        elif viewer == 'ds9':
            raise NotImplementedError

    def sky_image_integral(self, emin, emax, nbins=10, per_decade=False, interpolation='linear'):
        """
        Integrate cube along the energy axes using the log-log trapezoidal rule.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Integration range minimum.
        emax : `~astropy.units.Quantity`
            Integration range maximum.
        nbins : int, optional
            Number of grid points used for the integration.
        per_decade : bool
            Whether nbins is per decade.
        interpolation : {None, 'linear', 'nearest'}
            Interpolate data values between energies.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            Integral image.
        """
        y, x = np.indices(self.data.shape[1:])

        if interpolation:
            energy = Energy.equal_log_spacing(emin, emax, nbins, per_decade=per_decade)
            z = self.energy_axis.world2pix(energy).reshape(-1, 1, 1)
            y = np.arange(self.data.shape[1])
            x = np.arange(self.data.shape[2])
            z, y, x = np.meshgrid(z, y, x, indexing='ij')
            data = self._interpolate_data(z, y, x)
        else:
            energy = self.energy_axis.energy
            data = self.data
        integral = _trapz_loglog(data, energy, axis=0)
        name = 'integrated {}'.format(self.name)
        return SkyImage(name=name, data=integral, wcs=self.wcs.celestial)


    def reproject(self, reference, mode='interp', *args, **kwargs):
        """Spatially reprojects a `SkyCube` onto a reference.

        Parameters
        ----------
        reference : `~astropy.io.fits.Header`, `SkyImage` or `SkyCube`
            Reference wcs specification to reproject the data on.
        mode : {'interp', 'exact'}
            Interpolation mode.
        *args : list
            Arguments passed to `~reproject.reproject_interp` or
            `~reproject.reproject_exact`.
        **kwargs : dict
            Keyword arguments passed to `~reproject.reproject_interp` or
            `~reproject.reproject_exact`.

        Returns
        -------
        reprojected_cube : `SkyCube`
            Cube spatially reprojected to the reference.
        """
        if isinstance(reference, SkyCube):
            reference = reference.ref_sky_image

        out = []
        for idx in range(len(self.data)):
            image = self.sky_image(idx)
            image_out = image.reproject(reference, mode=mode, *args, **kwargs)
            out.append(image_out.data)

        data = Quantity(np.stack(out, axis=0), self.data.unit)
        wcs = image_out.wcs.copy()
        return self.__class__(name=self.name, data=data, wcs=wcs, meta=self.meta,
                              energy=self.energy_axis.energy)


    def to_fits(self):
        """Writes SkyCube to FITS hdu_list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            * hdu_list[0] : `~astropy.io.fits.PrimaryHDU`
                Image array of data
            * hdu_list[1] : `~astropy.io.fits.BinTableHDU`
                Table of energies
        """
        image = fits.PrimaryHDU(self.data.value, self.wcs.to_header())
        image.header['SPECUNIT'] = '{0.unit:FITS}'.format(self.data)

        # for BinTableHDU's the data must be added via a Table object
        energy_table = Table()
        energy_table['Energy'] = self.energy_axis.energy
        energy_table.meta['name'] = 'ENERGY'

        energies = table_to_fits_table(energy_table)

        hdu_list = fits.HDUList([image, energies])

        return hdu_list

    def to_images(self):
        """Convert to `~gammapy.cube.SkyCubeImages`.
        """
        from .images import SkyCubeImages
        images = [self.sky_image(energy) for energy in self.energy_axis.energy]
        return SkyCubeImages(self.name, images, self.wcs, self.energy_axis.energy)

    def write(self, filename, **kwargs):
        """Write to FITS file.

        Parameters
        ----------
        filename : str
            Filename
        """
        filename = str(make_path(filename))
        self.to_fits().writeto(filename, **kwargs)

    def __repr__(self):
        # Copied from `spectral-cube` package
        ss = "Sky cube {} with shape={}".format(self.name, self.data.shape)
        if self.data.unit is u.dimensionless_unscaled:
            ss += ":\n"
        else:
            ss += " and unit={}:\n".format(self.data.unit)

        ss += " n_lon:    {:5d}  type_lon:    {:15s}  unit_lon:    {}\n".format(
            self.data.shape[2], self.wcs.wcs.ctype[0], self.wcs.wcs.cunit[0])
        ss += " n_lat:    {:5d}  type_lat:    {:15s}  unit_lat:    {}\n".format(
            self.data.shape[1], self.wcs.wcs.ctype[1], self.wcs.wcs.cunit[1])
        ss += " n_energy: {:5d}  unit_energy: {}\n".format(
            self.data.shape[0], self.energy_axis.energy.unit)

        return ss

    def info(self):
        """
        Print summary info about the cube.
        """
        print(repr(self))

    @staticmethod
    def assert_allclose(cube1, cube2):
        assert cube1.name == cube2.name
        assert_allclose(cube1.data, cube2.data)
        assert_wcs_allclose(cube1.wcs, cube2.wcs)
