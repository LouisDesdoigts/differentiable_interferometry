import jax
import jax.numpy as np
import jax.scipy as jsp
import dLux as dl
import dLux.utils as dlu
import equinox as eqx
from jax.scipy.signal import convolve
from matplotlib import colormaps
from jax import vmap

import optax


# TODO: Add an optional 'norm_fn' that normalises the model as desired
# Maybe a process_grads fn too for fixing params?
def fit_image(
    # tel,
    model,
    data,
    err,
    # file,
    loss_fn,
    grad_fn,
    norm_fn,
    epochs,
    config,
    loss_scale=1e-4,
    verbose=True,
    print_grads=False,
):
    params = list(config.keys())
    optimisers = [config[param]["optim"] for param in params]

    model = zdx.set_array(model, params)
    optim, opt_state = zdx.get_optimiser(model, params, optimisers)

    if verbose:
        print("Compiling...")
    loss, grads = loss_fn(model, data, err)
    if print_grads:
        for param in params:
            print(f"{param}: {grads.get(param)}")
    losses, models_out = [], [model]

    if verbose:
        looper = tqdm(range(epochs), desc="Loss %.2f" % (loss * loss_scale))
    else:
        looper = range(epochs)

    for i in looper:
        # calculate the loss and gradient
        new_loss, grads = loss_fn(model, data, err)

        if new_loss > loss:
            print(
                f"Loss increased from {loss * loss_scale:.2f} to "
                f"{new_loss * loss_scale:.2f} on {i} th epoch"
            )
        loss = new_loss
        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch")
            return losses, models_out

        # Apply any processing to the gradients
        grads = grad_fn(grads, config, i)

        # apply the update
        updates, opt_state = optim.update(grads, opt_state)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model)

        # save results
        models_out.append(model)
        losses.append(loss)

        if verbose:
            looper.set_description("Loss %.2f" % (loss * loss_scale))

    return losses, models_out


def planck(wav, T):
    h = 6.626e-34
    c = 3.0e8
    k = 1.38e-23
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity


### Sub-propagations ###
def transfer(coords, npixels, wavelength, pscale, distance):
    """
    The optical transfer function (OTF) for the gaussian beam.
    Assumes propagation is along the axis.
    """
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)
    return np.exp(-1.0j * np.pi * wavelength * distance * rho_sq)


def _fft(phasor):
    return 1 / phasor.shape[0] * np.fft.fft2(phasor)


def _ifft(phasor):
    return phasor.shape[0] * np.fft.ifft2(phasor)


def plane_to_plane(wf, distance):
    tf = transfer(wf.coordinates, wf.npixels, wf.wavelength, wf.pixel_scale, distance)
    phasor = _fft(wf.phasor)
    phasor *= np.fft.fftshift(tf)
    phasor = _ifft(phasor)
    return phasor


class FreeSpace(dl.layers.optics.OpticalLayer):
    distance: jax.Array

    def __init__(self, dist):
        self.distance = np.asarray(dist, float)

    def apply(self, wf):
        phasor_out = plane_to_plane(wf, self.distance)
        return wf.set(
            ["amplitude", "phase"], [np.abs(phasor_out), np.angle(phasor_out)]
        )


class PupilAmplitudes(dl.layers.optics.OpticalLayer):
    basis: jax.Array
    coefficients: jax.Array

    def __init__(self, basis, coefficients=None):
        self.basis = np.asarray(basis, float)

        if coefficients is None:
            self.coefficients = np.zeros(basis.shape[:-2])
        else:
            self.coefficients = np.asarray(coefficients, float)

    def normalise(self):
        # Normalise to mean of 1
        return self.add("coefficients", self.coefficients.mean())

    def apply(self, wavefront):
        self = self.normalise()
        amplitudes = 1 + dlu.eval_basis(self.basis, self.coefficients)
        return wavefront.multiply("amplitude", amplitudes)


from typing import List, Any


class FresnelOptics(dl.CartesianOpticalSystem):
    """
    fl = pixel_scale_m / pixel_scale_rad -> NIRISS pixel scales are 18um  and
    0.0656 arcsec respectively, so fl ~= 56.6m
    """

    defocus: jax.Array  # metres, is this actually um??

    def __init__(self, *args, **kwargs):
        self.defocus = np.array(0.0)
        super().__init__(*args, **kwargs)

    def propagate_mono(
        self: dl.optical_systems.OpticalSystem,
        wavelength: jax.Array,
        offset: jax.Array = np.zeros(2),
        return_wf: bool = False,
    ) -> jax.Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        # Unintuitive syntax here, this is saying call the _parent class_ of
        # CartesianOpticalSystem, ie LayeredOpticalSystem, which is what we want.
        wf = super(dl.optical_systems.CartesianOpticalSystem, self).propagate_mono(
            wavelength, offset, return_wf=True
        )

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample

        wf = wf.propagate_fresnel(
            psf_npixels,
            pixel_scale,
            self.focal_length,
            focal_shift=self.defocus,
        )

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

inferno = colormaps["inferno"]
seismic = colormaps["seismic"]


def dsamp(arr, k):
    return arr.reshape(-1, k).mean(1)


def osamp_freqs(n, dx, osamp):
    df = 1 / (n * dx)
    odf = df / osamp

    if n % 2 == 0:
        start = -1 / (2 * dx)
        end = (n - 2) / (2 * dx * n)
    else:
        start = (1 - n) / (2 * n * dx)
        end = (n - 1) / (2 * n * dx)

    ostart = start + (odf - df) / 2
    oend = end + (df - odf) / 2
    return np.linspace(ostart, oend, n * osamp, endpoint=True)


# ns = [1, 2, 3, 4, 5, 6, 7]
# dxs = [1, 2, 3, 4, 5]
# ps = [1, 2, 3, 4, 5]

# TODO: Make this a test
# for n in ns:
#     for dx in dxs:
#         for p in ps:
#             base_freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
#             freqs = osamp_freqs(n, dx, p)
#             print(n, dx, np.isclose(base_freqs, dsamp(freqs, p)))


# import numpy as np


def pairwise_vectors(points):
    """
    Generates a non-redundant list of the pairwise vectors connecting each point in an array of (x,y) points,
    ordered ascendingly by the length of the vector.

    Args:
        points (ndarray): An array of shape (n, 2) containing the (x,y) coordinates of the points.

    Returns:
        list: A list of tuples containing the pairwise vectors connecting each point, ordered ascendingly by the length of the vector.
    """
    # Compute the pairwise vectors between each point
    vectors = points[:, np.newaxis] - points

    # Compute the lengths of the pairwise vectors
    lengths = np.sqrt(np.sum(vectors**2, axis=-1))

    # Create a list of non-redundant pairwise vectors
    pairwise_vectors = []
    for i in range(vectors.shape[0]):
        for j in range(i + 1, vectors.shape[1]):
            pairwise_vectors.append((vectors[i, j], i, j))

    # Sort the pairwise vectors by length
    # TODO: Replace with an argsort?
    pairwise_vectors.sort(key=lambda x: lengths[x[1], x[2]])
    # return pairwise_vectors

    vecs = []
    for vec in pairwise_vectors:
        v = vec[0]
        vecs.append([v[1], v[0]])
    return np.array(vecs)


# def hex_from_bls(coords, bl, rmax):
def hex_from_bls(bl, coords, rmax):
    # coords = dlu.translate_coords(coords, np.array([-bl[0], bl[1]]))
    coords = dlu.translate_coords(coords, np.array(bl))
    return dlu.reg_polygon(coords, rmax, 6)


def get_baselines(holes):
    # Get the baselines in m/wavelength (I do not know how this works)
    hole_mask = np.where(~np.eye(holes.shape[0], dtype=bool))
    thisu = (holes[:, 0, None] - holes[None, :, 0]).T[hole_mask]
    thisv = (holes[:, 1, None] - holes[None, :, 1]).T[hole_mask]
    return np.array([thisv, thisu]).T


def uv_hex_mask(
    holes,
    f2f,
    wavelength,
    psf_pscale,
    psf_npix,
    psf_oversample,
    uv_pad,
    mask_pad,
    verbose=False,
):
    """
    Holes: Hole positions, meters
    f2f: Hexagonal hole flat to flat distance, meters
    wavelength: Wavelength, meters
    psf_pscale: psf pixel scale, arcsec/pix
    psf_npix: psf npixels
    psf_oversample: oversampling of the psf
    uv_pad: padding before transforming to the uv plane
    mask_pad: mask calculation padding (ie mask oversample)
    """
    psf_npix *= psf_oversample

    # Correctly centred over sampled corods
    dx = dlu.arcsec2rad(psf_pscale) / psf_oversample
    shifted_coords = osamp_freqs(psf_npix * uv_pad, dx, mask_pad)
    uv_coords = np.array(np.meshgrid(shifted_coords, shifted_coords))

    # Do this outside so we can scatter plot the baseline vectors over the psf splodges
    hbls = pairwise_vectors(holes) / wavelength

    # Hole parameters
    rmax = f2f / np.sqrt(3)
    rmax_in = 2 * rmax / wavelength  # x2 because size doubles through a correlation

    # Get splodge masks and append DC term
    uv_hexes = []
    uv_hexes_conj = []

    # Baselines
    if verbose:
        looper = tqdm(hbls)
    else:
        looper = hbls

    for bl in looper:
        uv_hexes.append(hex_from_bls(bl, uv_coords, rmax_in))
        uv_hexes_conj.append(hex_from_bls(-1 * bl, uv_coords, rmax_in))
    uv_hexes = np.array(uv_hexes)
    uv_hexes_conj = np.array(uv_hexes_conj)

    dc_hex = np.array([hex_from_bls([0, 0], uv_coords, rmax_in)])

    hexes = np.concatenate([dc_hex, uv_hexes, uv_hexes_conj])

    # Normalise
    norm_hexes = dlu.nandiv(hexes, hexes.sum(0), 0.0)
    dsampler = vmap(lambda arr: dlu.downsample(arr, mask_pad))
    return dsampler(norm_hexes)


def compare_mask(cplx, masks, pad):
    ampl = np.abs(cplx)
    mask = masks.sum(0) > 0

    c = ampl.shape[0] // 2
    s = 30 * pad
    cut = slice(c - s, c + s, 1)
    ampl = ampl[cut, cut]
    mask = mask[cut, cut]

    inner = ampl * mask
    outer = ampl * np.abs(mask - 1)

    logged = np.where(np.log10(ampl) == -np.inf, np.nan, np.log10(ampl))
    vmin, vmax = np.nanmin(logged), np.nanmax(logged)

    # plt.figure(figsize=(20, 8))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(inner), vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(outer), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

    inner_mask_min = np.nanmin(np.log10(inner[np.where(mask)]))
    masked_outer = np.where(mask, -np.inf, np.log10(outer))
    nand_outer = np.where(masked_outer == -np.inf, np.nan, masked_outer)

    outer_mask_max = np.nanmax(nand_outer)
    inner_mask_min = np.nanmin(np.log10(inner[np.where(mask)]))
    print(f"Inner min: {inner_mask_min}")
    print(f"Outer max: {outer_mask_max}")
    print(f"Inner - outer: {inner_mask_min - outer_mask_max}")


class UVSource(dl.BaseSource):
    wavelengths: jax.Array
    weights: jax.Array
    position: jax.Array  # arcsec
    flux: jax.Array
    mask: jax.Array
    amplitudes: jax.Array
    phases: jax.Array
    pad: int

    def __init__(
        self,
        wavelengths,  # Wavelengths in meters
        mask,
        position=np.zeros(2),  # Source position in arcsec
        flux=1,  # Source Flux
        pad=2,  # UV-transform padding factor
        weights=None,  # Spectral weights
    ):
        """
        Assumes the last term in the mask is the DC term, and that the first half of the
        array is the positive frequencies and the second half is the negative baselines.
        """
        # Set up wavelengths and weights
        self.wavelengths = np.asarray(wavelengths, float)
        if weights is None:
            weights = np.ones_like(self.wavelengths)
        self.weights = np.asarray(weights, float)

        # Ensure wavelengths and weights are the same shape
        if self.wavelengths.shape != self.weights.shape:
            raise ValueError(
                f"Shape mismatch between wavelengths ({self.wavelengths.shape}) and "
                f"weights ({self.weights.shape})"
            )

        # Set up position and flux
        self.position = np.asarray(position, float)
        self.flux = np.asarray(flux, float)
        if self.position.shape != (2,):
            raise ValueError(
                f"Position must be a 2-element array, not {self.position.shape}"
            )

        self.pad = int(pad)
        self.mask = mask

        # Construct amplitudes and phases
        N = (self.mask.shape[1] - 1) // 2  # Should always be even after -1, because DC
        # self.amplitudes = np.ones(N + 1)  # +1 for dc term
        # self.phases = np.zeros(N + 1)  # +1 for dc term
        self.amplitudes = np.ones(N)
        self.phases = np.zeros(N)

    def _to_uv(self, psf):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))

    def _from_uv(self, uv):
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uv)))

    # TODO: Rename apply visibilites, take splodge and inv from self
    def _apply_splodge(self, psf, splodge, inv_splodge_support):
        cplx = self._to_uv(psf)
        splodged_cplx = cplx * splodge
        inv_splodge_cplx = cplx * inv_splodge_support
        return self._from_uv(splodged_cplx + inv_splodge_cplx)

    @property
    def visibilities(self):
        return self.amplitudes * np.exp(1j * self.phases)

    def normalise(self):
        return self.divide("weights", self.weights.sum())
        # norm_weights = self.weights / self.weights.sum()
        # norm_phases = self.phases - self.phases[0]
        # norm_phases = self.phases
        # norm_ampls = 1 + self.amplitudes - self.amplitudes[0]
        # norm_ampls = self.amplitudes / self.amplitudes[0]
        # norm_ampls = self.amplitudes
        # return self.set(
        # ["weights", "phases", "amplitudes"], [norm_weights, norm_phases, norm_ampls]
        # )

    @property
    def N(self):
        # return len(self.amplitudes) - 1  # -1 for dc term
        return len(self.amplitudes)

    @property
    def splodges(self):
        # Get the components of the calculation
        dc_mask = self.mask[:, 0]
        mask = self.mask[:, 1 : self.N + 1]
        conj_mask = self.mask[:, -self.N :]
        vis = self.visibilities

        # Get the splodges
        dot = lambda a, b: dlu.eval_basis(a, b)
        splodge_fn = lambda dc_mask, mask, conj_mask: (
            # dc_mask * vis[0] + dot(mask, vis[1:]) + dot(conj_mask, vis[1:].conj())
            dc_mask
            + dot(mask, vis)
            + dot(conj_mask, vis.conj())
        )
        return vmap(splodge_fn)(dc_mask, mask, conj_mask)

    @property
    def inv_splodges_support(self):
        return np.abs(1 - self.mask.sum(1))

    # TODO: Allow for return wf and psf
    def model(self, optics, return_wf=False, return_psf=False):
        """ """
        # Normalise
        self = self.normalise()

        # Calculate the PSF
        pos_rad = dlu.arcsec2rad(self.position)
        wgts = self.flux * self.weights
        wfs = optics.propagate(self.wavelengths, pos_rad, wgts, return_wf=True)
        psfs = wfs.psf

        # Apply padding
        npix = self.pad * psfs[0].shape[0]
        padded = vmap(lambda x: dlu.resize(x, npix))(psfs)

        # Shape check
        if padded[0].shape != self.mask.shape[-2:]:
            raise ValueError(
                f"PSF shape {padded[0].shape} does not match mask shape "
                f"{self.mask.shape}. This is likely because the wrong `npix` or "
                "`oversample` value was provided to the constructor."
            )

        splodges = self.splodges
        inv_splodges_support = self.inv_splodges_support
        cplx_psfs = []
        for i in range(len(padded)):
            cplx_psfs.append(
                self._apply_splodge(padded[i], splodges[i], inv_splodges_support[i])
            )
        cplx_psfs = np.array(cplx_psfs)
        cplx_psfs = vmap(lambda x: dlu.resize(x, psfs.shape[-1]))(cplx_psfs)

        # Return wf
        if return_wf:
            return wfs.set(
                ["amplitude", "phase"], [np.abs(cplx_psfs), np.angle(cplx_psfs)]
            )
        if return_psf:
            return eqx.filter_vmap(dl.PSF)(np.abs(cplx_psfs), wfs.pixel_scale)

        return np.abs(cplx_psfs).sum(0)


from jax.scipy.ndimage import map_coordinates


def arr2pix(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return pscale * (coords - shift)


def pix2arr(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return (coords / pscale) + shift


class Interpolator(dl.layers.unified_layers.UnifiedLayer):
    transform: dl.CoordTransform
    order: int
    layer: None

    def __init__(self, layer, transform, order=1):
        self.transform = transform
        self.layer = layer
        self.order = int(order)

    # def __getattribute__(self, name):
    #     output = super().__getattribute__(name)
    #     if self._check(output):
    #         print("getattribute")
    #         return self.interpolate(output)
    #     return output

    def __getattr__(self, key):
        if hasattr(self.layer, key):
            # output = getattr(self.layer, key)
            # if self._check(output):
            #     print("getattr")
            #     return self.interpolate(output)
            # else:
            #     return output
            return getattr(self.layer, key)
        elif hasattr(self.transform, key):
            return getattr(self.transform, key)
        else:
            raise AttributeError(f"Interpolator has no attribute {key}")

    def _check(self, item):
        if (
            isinstance(item, jax.Array)  # Array check
            and item.ndim == 2  # 2D check
            and item.shape[0] == item.shape[1]  # Square check
        ):
            return True
        return False

    def orig_coords(self, arr):
        # Generate paraxial coords with pixel scale of 1
        return dlu.pixel_coords(arr.shape[0], arr.shape[0])

    def interp_coords(self, arr):
        # Apply the transformation
        return self.transform.apply(self.orig_coords(arr))

    def pix_coords(self, arr):
        # Convert from pixel to array coords
        coords = pix2arr(self.interp_coords(arr))

        # indexing convention swap: (x, y) -> (i, j)
        return np.array([coords[1], coords[0]])

    def interpolate(self, arr):
        return map_coordinates(arr, self.pix_coords(arr), 1)

    @property
    def transformed(self):
        fn = lambda leaf: self.interpolate(leaf) if self._check(leaf) else leaf
        return jax.tree_util.tree_map(fn, self.layer)

    def apply(self, wavefront):
        return self.transformed.apply(wavefront)


# tf = dl.CoordTransform((0.0, 0.0), 0.0, (1.0, 1.0), (0.0, 0.0))
# _tel = tel.set("pupil_mask", Interpolator(tel.pupil_mask, tf))


import scipy
import numpy as onp


def UD_vis2(wavel, diam, u=None, v=None, baseline=None):
    """
    Calculate the squared visibility at a given u, v coord
    or baseline for a uniform disk source morphology.

    Parameters
    ----------
    wavel : float, nanometres
        Wavelength in nm.
    diam : float, radians
        Diameter of the uniform disk in radians.
    u : float, m
        u coordinate in meters.
    v : float, m
        v coordinate in meters.
    baseline : float, m
        Baseline length in meters. If specified u and v are ignored.

    """
    if baseline is None:
        baseline = np.sqrt(u**2 + v**2)  # grabbing baseline length

    wavel *= 1e-9  # nm -> m
    A = np.pi * diam * baseline / wavel
    return (2 * scipy.special.j1(A) / A) ** 2


def GD_vis2(wavel, sigma, u=None, v=None, baseline=None):
    """
    Calculate the squared visibility at a given u, v coord
    or baseline for a uniform disk source morphology.

    Parameters
    ----------
    wavel : float, nanometres
        Wavelength in nm.
    sigma : float, radians
        Standard deviation of the Gaussian disk in radians.
    u : float, m
        u coordinate in meters.
    v : float, m
        v coordinate in meters.
    baseline : float, m
        Baseline length in meters. If specified u and v are ignored.

    Returns
    -------
    vis2 : float
        Squared visibility.
    """
    if baseline is None:
        baseline = np.sqrt(u**2 + v**2)  # grabbing baseline length
    FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma  # FWHM in radians
    wavel *= 1e-9  # nm -> m

    return np.exp(-((np.pi * FWHM * baseline / wavel) ** 2) / (2 * np.log(2)))


def binary_vis2(u, v, wavel, sep, pa, flux_ratio=1, baseline=None):
    """
    Calculate the squared visibility at a given u, v coord
    or baseline for a resolved binary pair.

    See Table 1
    https://ui.adsabs.harvard.edu/abs/2007NewAR..51..576B

    Parameters
    ----------
    u : float, m
        u coordinate in meters.
    v : float, m
        v coordinate in meters.
    wavel : float, nanometres
        Wavelength in nm.
    sep : float, degrees
        Angular separation of the binary pair in radians.
    pa : float, radians
        Position angle of the binary pair in radians.
    flux_ratio : float = 1
        Ratio of the flux of the secondary to the primary. Must be between (0, 1].

    Returns
    -------
    vis2 : float
        Squared visibility.
    """
    if flux_ratio > 1 or flux_ratio <= 0:
        raise ValueError("flux_ratio must be between (0, 1].")

    if baseline is None:
        baseline = np.sqrt(u**2 + v**2)  # grabbing baseline length

    wavel *= 1e-9  # nm -> m

    # angular separation vector in radians
    rho = np.array([sep * np.sin(pa), sep * np.cos(pa)])

    # dot product between baseline vector and angular separation vector
    dot = np.dot(np.array([u, v]), rho)

    return (
        1 + flux_ratio**2 + 2 * flux_ratio * np.cos(2 * np.pi / (wavel * dot))
    ) / (1 + flux_ratio) ** 2


def summarise_files(files, extra_keys=[]):
    main_keys = [
        "TARGPROP",
        # "TARGNAME",
        "FILTER",
        "OBSERVTN",
        "PATTTYPE",
        # "APERNAME",
        # "PUPIL",
        # "SUBARRAY",
        # "DETECTOR",
        # "DATAMODL",
        # "INSTRUME",
        # "EXP_TYPE",
    ]

    main_keys += extra_keys
    for key in main_keys:
        values = set([f[0].header[key] for f in files])
        vals_str = ", ".join([f"{val}" for val in values])
        print(f"  {key}: {vals_str}")


def get_files(data_path, ext, **kwargs):
    """

    data_path: Path to the data files
    ext: File extension to search for
    """
    import os
    from astropy.io import fits

    file_names = os.listdir(data_path)

    files = []
    checked = False
    for name in file_names:
        if name.endswith(f"{ext}.fits"):
            file = fits.open(data_path + name)
            h = file[0].header

            if not checked:
                if not all([key in h.keys() for key in kwargs.keys()]):
                    raise KeyError(
                        f"Header keys {kwargs.keys()} not found in file {name}"
                    )

            match = True
            for key, val in kwargs.items():
                if isinstance(val, list):
                    if h[key] not in val:
                        match = False
                elif h[key] != val:
                    match = False

            if match:
                files.append(file)
    return files


def get_webb_osys_fits(file):
    import webbpsf
    import datetime

    inst = getattr(webbpsf, file[0].header["INSTRUME"])()

    # Set filter
    inst.filter = file[0].header["FILTER"]

    # Set aperture
    inst.aperturename = file[0].header["APERNAME"]

    # Set pupil mask
    if file[0].header["PUPIL"] == "NRM":
        pupil_in = "MASK_NRM"
    else:
        pupil_in = file[0].header["PUPIL"]
    inst.pupil_mask = pupil_in

    # Set WFS data
    d1 = datetime.datetime.fromisoformat(file[0].header["DATE-BEG"])
    d2 = datetime.datetime.fromisoformat(file[0].header["DATE-END"])

    # Weirdness here because you cant add datetimes
    time = (d1 + (d2 - d1) / 2).isoformat()

    # Load WFS data
    print("Loading WFS data")
    inst.load_wss_opd_by_date(time, verbose=False)

    # Calculate data to ensure things are populated correctly
    psf_fits = inst.calc_psf()

    return inst, psf_fits


def initialise_for_data(tel, file):
    psf_npix, pos, flux = get_intial_values(tel, file)
    return tel.set(["psf_npixels", "position", "flux"], [psf_npix, pos, flux])


def get_intial_values(tel, file):
    # Enforce correct npix
    im = np.array(file[1].data).astype(float)
    im = im.at[np.where(np.isnan(im))].set(0.0)

    # Get naive model
    tel = tel.set("psf_npixels", im.shape[0])
    tel = tel.set("position", np.zeros(2))
    tel = tel.set("flux", im.sum())
    psf = tel.model()

    # Get position
    conv = convolve(im, psf, mode="same")

    max_idx = np.array(np.where(conv == conv.max())).squeeze()
    parax_pix_pos = max_idx - im.shape[0] // 2

    if not isinstance(tel.optics, (dl.AngularOpticalSystem, dl.CartesianOpticalSystem)):
        optics = tel.optics.optics
    else:
        optics = tel.optics

    if isinstance(optics, dl.CartesianOpticalSystem):
        pscale = dlu.rad2arcsec(1e-6 * optics.psf_pixel_scale / optics.focal_length)
    elif isinstance(optics, dl.AngularOpticalSystem):
        pscale = optics.psf_pixel_scale
    else:
        raise ValueError("Optics must be Cartesian or Angular")
    pos = np.roll(pscale * parax_pix_pos, 1)
    pos *= np.array([1, -1])

    # Get flux
    ratio = im.sum() / psf.sum()
    flux = ratio * im.sum()

    return im.shape[0], pos, flux


def get_AMI_splodge_mask(tel, wavelengths, calc_pad=2, pad=2, verbose=True, f2f=0.82):
    from nrm_analysis.misctools import mask_definitions

    # Take holes from ImPlaneIA
    holes = mask_definitions.jwst_g7s6c()[1]
    # f2f = 0.82  # m

    # Get values from telescope
    oversample = tel.oversample
    psf_npix = tel.psf_npixels
    pscale = tel.psf_pixel_scale

    if verbose:
        looper = tqdm(wavelengths)
    else:
        looper = wavelengths

    # Now we calculate the masks
    masks = []
    for wl in looper:
        masks.append(
            uv_hex_mask(holes, f2f, wl, pscale, psf_npix, oversample, pad, calc_pad)
        )

    return np.array(masks)


def convert_adjacent_to_true(bool_array):
    trues = np.array(np.where(bool_array))
    trues = np.swapaxes(trues, 0, 1)
    for i in range(len(trues)):
        y, x = trues[i]
        bool_array = bool_array.at[y, x + 1].set(True)
        bool_array = bool_array.at[y, x - 1].set(True)
        bool_array = bool_array.at[y + 1, x].set(True)
        bool_array = bool_array.at[y - 1, x].set(True)
    return bool_array


def nan_brightest(array, n_mask, order=1, thresh=None):
    # Get the high flux mask
    if n_mask > 0:
        sorted = np.sort(array.flatten())
        thresh_in = sorted[~np.isnan(sorted)][-n_mask]

        if thresh is not None:
            thresh_in = np.minimum(thresh, thresh_in)

        flux_mask = convert_adjacent_to_true(array >= thresh_in)
        if order > 1:
            for i in range(order - 1):
                flux_mask = convert_adjacent_to_true(flux_mask)
        return array.at[np.where(flux_mask)].set(np.nan)


def get_nan_support(file, n_mask=1, order=1, thresh=None):
    # Get the data we need
    im = np.array(file[1].data).astype(float)
    dq = np.array(file[3].data).astype(bool)

    im = nan_brightest(im, n_mask, order, thresh)
    return ~np.isnan(im) & ~dq


# def gettr(im, support):
#     return im[support[0], support[1]]


# def like_fn(model, data, sigma, sup):
#     return jsp.stats.norm.pdf(
#         gettr(model.model(), sup), loc=gettr(data, sup), scale=gettr(sigma, sup)
#     )


# def loglike_fn(model, data, sigma, sup):
#     return jsp.stats.norm.logpdf(
#         gettr(model.model(), sup), loc=gettr(data, sup), scale=gettr(sigma, sup)
#     )
#     # return jsp.stats.norm.logpdf(model.model(), loc=data, scale=sigma)


def get_likelihoods(psf, data, err):
    return (
        jsp.stats.norm.pdf(psf, loc=data, scale=err),
        -jsp.stats.norm.logpdf(psf, loc=data, scale=err),
    )


# def show_likelihoods(model, file, show_res=True, n_mask=1, order=1, k=0.5):
#     im = np.array(file[1].data).astype(float)
#     err = np.array(file[2].data).astype(float)
#     support, support_mask = get_nan_support(file, n_mask=n_mask, order=order)

#     like_px = like_fn(model, im, err, support)  # pixel likelihood 1d
#     loglike_px = -loglike_fn(model, im, err, support)  # pixel likelihood, 1d

#     like_im = np.ones_like(im) * np.nan
#     like_im = like_im.at[support[0], support[1]].set(like_px)
#     loglike_im = like_im.at[support[0], support[1]].set(loglike_px)

#     return like_im, loglike_im, support_mask

# return like_im, loglike_im

# if show_res:
#     psf = model.model().at[~support_mask].set(np.nan)
#     data = im.at[~support_mask].set(np.nan)
#     res = data - psf
#     n = 3
# else:
#     n = 2

# inferno.set_bad("k", k)
# plt.figure(figsize=(n * 5, 4))
# plt.subplot(1, n, 1)
# plt.title("Pixel likelihood")
# plt.imshow(like_im, cmap=inferno)
# plt.colorbar()

# plt.subplot(1, n, 2)
# plt.title("Pixel neg log likelihood")
# plt.imshow(loglike_im, cmap=inferno)
# plt.colorbar()

# if show_res:
#     seismic.set_bad("k", k)
#     v = np.nanmax(np.abs(res))
#     plt.subplot(1, n, 3)
#     plt.title("Residual")
#     plt.imshow(res, cmap=seismic, vmin=-v, vmax=v)
#     plt.colorbar()

# plt.tight_layout()
# plt.show()


def plot_image(fits_file, idx=0):
    def getter(file, k):
        if file[k].data.ndim == 2:
            return file[k].data
        else:
            return file[k].data[idx]

    plt.figure(figsize=(15, 4))

    plt.suptitle(fits_file[1].data.shape)

    plt.subplot(1, 3, 1)
    plt.title("SCI")
    plt.imshow(getter(fits_file, 1))
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("ERR")
    plt.imshow(getter(fits_file, 2))
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("DQ")
    plt.imshow(getter(fits_file, 3))
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def compare(arr1, arr2, cut=0, normres=True, titles=["arr1", "arr2"], k=0.5):
    arr1 = np.array(arr1).astype(float)
    arr2 = np.array(arr2).astype(float)

    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError("Arrays must be 2D")

    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have same shape")

    c = arr1.shape[0] // 2
    s = c - cut
    cut = slice(c - s, c + s, 1)

    res = arr1 - arr2

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title(titles[0])
    plt.imshow(arr1[cut, cut])
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title(titles[1])
    plt.imshow(arr2[cut, cut])
    plt.colorbar()

    v = np.nanmax(np.abs(res[cut, cut]))
    seismic.set_bad("k", k)
    plt.subplot(1, 3, 3)
    plt.title("Residual")
    plt.imshow(res[cut, cut], cmap=seismic, vmin=-v, vmax=v)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# def plot_vis(holes, model, fim):
def plot_vis(holes, model, fim=None):
    if fim is not None:
        N = len(model.amplitudes)
        stds = np.abs(np.diag(-np.linalg.inv(fim))) ** 0.5
        ampl_sig = stds[-2 * N : -N]
        phase_sig = stds[-N:]

    hbls = pairwise_vectors(holes)
    bls_r = np.array(np.hypot(hbls[:, 0], hbls[:, 1]))
    # bls_r = np.concatenate([np.zeros(1), bls_r])  # Add DC term

    plt.figure(figsize=(14, 5))
    plt.suptitle("Visibilities")

    plt.subplot(1, 2, 1)
    if fim is not None:
        plt.title(f"Amplitudes, mean sigma: {ampl_sig.mean():.3f}")
        plt.errorbar(bls_r, model.amplitudes, yerr=ampl_sig, fmt="o", capsize=5)
    else:
        plt.title("Amplitudes")
        plt.scatter(bls_r, model.amplitudes)
    plt.axhline(1, c="k", ls="--")
    plt.ylabel("Amplitudes")
    plt.xlabel("Baseline (m)")

    plt.subplot(1, 2, 2)
    if fim is not None:
        plt.title(f"Phases, mean sigma: {phase_sig.mean():.3f}")
        plt.errorbar(bls_r, model.phases, yerr=phase_sig, fmt="o", capsize=5)
    else:
        plt.title("Phases")
        plt.scatter(bls_r, model.phases)
    plt.axhline(0, c="k", ls="--")
    plt.ylabel("Phases")
    plt.xlabel("Baseline (m)")

    plt.tight_layout()
    plt.show()


def show_splodges(model, s=65, pupil_phases=False, k=0.5):
    splodges = model.source.splodges

    c = splodges.shape[-1] // 2
    cut = slice(c - s, c + s, 1)

    # Get the coordaintes
    dx = dlu.arcsec2rad(model.psf_pixel_scale) / model.oversample
    wl = model.wavelengths.mean()

    shifted_coords = osamp_freqs(splodges.shape[-1], dx, 1) * wl / 2
    rmin, rmax = shifted_coords[cut].min(), shifted_coords[cut].max()
    extent = [rmin, rmax, rmin, rmax]

    if pupil_phases:
        n = 3
    else:
        n = 2
    plt.figure(figsize=(n * 5, 4))

    plt.subplot(1, n, 1)
    plt.title("Applied Amplitudes")
    plt.imshow(np.abs(splodges).mean(0)[cut, cut], extent=extent)
    plt.colorbar()
    plt.xlabel("meters")
    plt.ylabel("meters")

    plt.subplot(1, n, 2)
    plt.title("Applied Phases")
    plt.imshow(np.angle(splodges).mean(0)[cut, cut], extent=extent, cmap="seismic")
    plt.colorbar(label="radians")
    plt.xlabel("meters")
    plt.ylabel("meters")

    if pupil_phases:
        seismic.set_bad("k", k)
        if hasattr(model.pupil_mask, "transformed"):
            transmisson = np.flipud(model.pupil_mask.transformed.transmission)
        else:
            transmisson = np.flipud(model.pupil_mask.transmission)
        opd = (model.basis_opd * 1e9).at[np.where(~(transmisson > 1e-6))].set(np.nan)
        v = np.nanmax(np.abs(opd))
        plt.subplot(1, n, 3)
        plt.title("Pupil Phases")
        plt.imshow(opd, extent=extent, vmin=-v, vmax=v, cmap=seismic)
        plt.colorbar(label="nm")
        plt.xlabel("meters")
        plt.ylabel("meters")

    plt.tight_layout()
    plt.show()


class Null(dl.layers.unified_layers.UnifiedLayer):
    def apply(self, inputs):
        return inputs


from jax import jit, grad, jvp, linearize, lax
import zodiax as zdx


def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


def hessian(f, x):
    _, hvp = linearize(grad(f), x)
    # Jit the sub-function here since it is called many timesc
    # TODO: Test effect on speed
    hvp = jit(hvp)
    basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    shape_dict={},
    save_ram=True,
    **loglike_kwargs,
):
    # Build X vec
    pytree = zdx.tree.set_array(pytree, parameters)
    shapes, lengths = zdx.bayes._shapes_and_lengths(pytree, parameters, shape_dict)
    X = np.zeros(zdx.bayes._lengths_to_N(lengths))

    # Build function to calculate FIM and calculate
    # @jax.hessian
    def calc_fim(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    if save_ram:
        return hessian(calc_fim, X)
    return jax.hessian(calc_fim)(X)


def _perturb(X, pytree, parameters, shapes, lengths):
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))

    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length - 1

    return pytree.add(parameters, xs)
