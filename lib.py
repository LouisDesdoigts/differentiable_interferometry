import jax
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
import equinox as eqx

from jax import vmap


from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


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
    # hexes = np.array(uv_hexes)

    hexes = np.concatenate([uv_hexes, uv_hexes_conj, dc_hex])

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
    # print(vmin, vmax)

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
        self.flux = np.array(flux)
        if self.position.shape != (2,):
            raise ValueError(
                f"Position must be a 2-element array, not {self.position.shape}"
            )

        self.pad = int(pad)
        self.mask = mask

        # Construct amplitudes and phases
        N = (self.mask.shape[1] - 1) // 2  # Should always be even after -1
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
        norm_weights = self.weights / self.weights.sum()
        norm_phases = self.phases - self.phases[0]
        return self.set(["weights", "phases"], [norm_weights, norm_phases])
        # norm_aml = self.amplitudes - (1 - self.dc)
        # return self.set(["weights", "amplitudes"], [norm_weights, norm_aml])
        # return self.divide("weights", self.weights.sum())

    @property
    def N(self):
        return len(self.amplitudes)

    @property
    def splodges(self):
        # Get the components of the calculation
        mask = self.mask[:, : self.N]
        conj_mask = self.mask[:, self.N : -1]
        # dc = self.dc * self.mask[:, -1]
        dc = self.mask[:, -1]
        vis = self.visibilities

        # Get the splodges
        dot = lambda a, b: dlu.eval_basis(a, b)
        splodge_fn = (
            lambda mask, conj_mask, dc: dot(mask, vis) + dot(conj_mask, vis.conj()) + dc
        )
        return vmap(splodge_fn)(mask, conj_mask, dc)

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

        # cplx_psfs = vmap(self._apply_splodge)(padded, self.splodges)
        # init_sum = np.abs(psfs).sum(0)
        # final_sum = np.abs(cplx_psfs).sum(0)
        # print(f"Initial sum: {init_sum.sum()}")
        # print(f"Final sum: {final_sum.sum()}")
        # print(f"Sum ratio: {final_sum.sum() / init_sum.sum()}")

        cplx_psfs = vmap(lambda x: dlu.resize(x, psfs.shape[-1]))(cplx_psfs)

        # Return wf
        if return_wf:
            return wfs.set(
                ["amplitude", "phase"], [np.abs(cplx_psfs), np.angle(cplx_psfs)]
            )
        if return_psf:
            return eqx.filter_vmap(dl.PSF)(np.abs(cplx_psfs), wfs.pixel_scale)

        return np.abs(cplx_psfs).sum(0)


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


# def binary_vis2(u, v, wavel, sep, pa, flux_ratio=1):
#     """
#     Calculate the squared visibility at a given u, v coord
#     or baseline for a resolved binary pair.

#     See Table 1
#     https://ui.adsabs.harvard.edu/abs/2007NewAR..51..576B

#     Parameters
#     ----------
#     u : float, m
#         u coordinate in meters.
#     v : float, m
#         v coordinate in meters.
#     wavel : float, nanometres
#         Wavelength in nm.
#     sep : float, degrees
#         Angular separation of the binary pair in radians.
#     pa : float, radians
#         Position angle of the binary pair in radians.
#     flux_ratio : float = 1
#         Ratio of the flux of the secondary to the primary. Must be between (0, 1].

#     Returns
#     -------
#     vis2 : float
#         Squared visibility.
#     """
#     if flux_ratio > 1 or flux_ratio <= 0:
#         raise ValueError("flux_ratio must be between (0, 1].")

#     if baseline is None:
#         baseline = np.sqrt(u**2 + v**2)  # grabbing baseline length

#     wavel *= 1e-9  # nm -> m

#     # angular separation vector in radians
#     rho = np.array([sep * np.sin(pa), sep * np.cos(pa)])

#     # dot product between baseline vector and angular separation vector
#     dot = np.dot(np.array([u, v]), rho)

#     return (
#         1 + flux_ratio**2 + 2 * flux_ratio * np.cos(2 * np.pi / (wavel * dot))
#     ) / (1 + flux_ratio) ** 2
