"""Synthetic degradation operators.

All operate on HWC float32 images in [0, 1] and return the same, so they compose
and can be sampled per-patch on the fly with no download.

Two families, chosen because they sit at opposite ends of the axis hypothesis:

* **noise** -- per-pixel corruption. Additive Gaussian is signal-INdependent and
  i.i.d., which is the case the channel-leaning prediction is built on. Speckle
  and Poisson are signal-DEPENDENT: their magnitude scales with local intensity,
  so they inherit spatial structure from the image. If the gate treats them
  differently from Gaussian, that is a finding, not a bug.

* **blur** -- spatial convolution. Gaussian and defocus are isotropic and compact
  (a few pixels); linear motion is anisotropic and can span tens of pixels. All
  three are spatial operators, so all should push the gate the same way; they
  differ in how much receptive field they demand.
"""
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# --------------------------------------------------------------------- noise
def add_gaussian(img, sigma, rng=None):
    """Additive white Gaussian noise. `sigma` on the 0-255 scale."""
    rng = rng or np.random
    n = rng.standard_normal(img.shape).astype(np.float32) if hasattr(rng, "standard_normal") \
        else rng.randn(*img.shape).astype(np.float32)
    return img + n * (sigma / 255.0)


def add_speckle(img, sigma, rng=None):
    """Multiplicative Gaussian speckle: J = I + I*n.

    Signal-dependent: bright regions get more noise than dark ones. This is the
    classic 'speckle' of ultrasound/SAR imagery in its Gaussian approximation.
    """
    rng = rng or np.random
    n = rng.standard_normal(img.shape).astype(np.float32) if hasattr(rng, "standard_normal") \
        else rng.randn(*img.shape).astype(np.float32)
    return img + img * n * (sigma / 255.0)


def add_speckle_gamma(img, looks=4.0, rng=None):
    """Fully-developed speckle: J = I * g, g ~ Gamma(L, 1/L), mean 1.

    The physically correct model for coherent imaging, and harsher than the
    Gaussian approximation at low `looks`. L -> infinity approaches no noise.
    """
    rng = rng or np.random.default_rng()
    g = rng.gamma(shape=looks, scale=1.0 / looks, size=img.shape).astype(np.float32)
    return img * g


def add_poisson(img, peak=30.0, rng=None):
    """Shot noise: J = Poisson(I * peak) / peak.

    Signal-dependent like speckle, but with variance equal to the mean rather
    than proportional to its square. Lower `peak` means more noise.
    """
    rng = rng or np.random.default_rng()
    return rng.poisson(np.clip(img, 0, 1) * peak).astype(np.float32) / peak


def add_salt_pepper(img, amount=0.02, rng=None):
    """Impulse noise. Included for completeness; not part of the core hypothesis."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0.0
    out[m > 1 - amount / 2] = 1.0
    return out


NOISE_OPS = {
    "gaussian": add_gaussian,
    "speckle": add_speckle,
    "speckle_gamma": add_speckle_gamma,
    "poisson": add_poisson,
    "salt_pepper": add_salt_pepper,
}


# ---------------------------------------------------------------------- blur
def gaussian_kernel(sigma, size=None):
    size = size or max(3, int(2 * round(3 * sigma) + 1))
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma * sigma))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


def defocus_kernel(radius):
    """Disk kernel -- the out-of-focus point spread function."""
    size = max(3, int(2 * round(radius) + 1))
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = ((xx ** 2 + yy ** 2) <= radius * radius).astype(np.float32)
    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else gaussian_kernel(1.0)


def motion_kernel(length, angle_deg):
    """Linear motion PSF: a line of `length` px at `angle_deg`."""
    length = max(int(round(length)), 1)
    size = length if length % 2 == 1 else length + 1
    k = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th), np.sin(th)
    for t in np.linspace(-length / 2.0, length / 2.0, length * 4):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else gaussian_kernel(1.0)


def apply_kernel(img, kernel):
    """Convolve HWC float image with a 2-D kernel, reflect-padded."""
    if _HAS_CV2:
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    pad = kernel.shape[0] // 2
    out = np.empty_like(img)
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    for c in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j, c] = np.sum(
                    padded[i:i + kernel.shape[0], j:j + kernel.shape[1], c] * kernel)
    return out


def sample_blur_kernel(spec, rng=None):
    """Build one kernel from a spec dict, e.g.
    {'kind': 'gaussian', 'sigma': [0.5, 3.0]} -- ranges are sampled uniformly.
    """
    rng = rng or np.random.default_rng()

    def pick(v):
        return rng.uniform(*v) if isinstance(v, (list, tuple)) else v

    kind = spec["kind"]
    if kind == "gaussian":
        return gaussian_kernel(pick(spec.get("sigma", [0.5, 3.0])))
    if kind == "defocus":
        return defocus_kernel(pick(spec.get("radius", [1.0, 5.0])))
    if kind == "motion":
        return motion_kernel(pick(spec.get("length", [5, 25])),
                             pick(spec.get("angle", [0, 180])))
    raise ValueError(f"unknown blur kind {kind!r}")


def apply_noise(img, spec, rng=None):
    """Apply one noise op from a spec dict, e.g.
    {'kind': 'gaussian', 'sigma': [5, 50]}.
    """
    rng = rng or np.random.default_rng()

    def pick(v):
        return rng.uniform(*v) if isinstance(v, (list, tuple)) else v

    kind = spec["kind"]
    if kind not in NOISE_OPS:
        raise ValueError(f"unknown noise kind {kind!r}; known: {sorted(NOISE_OPS)}")
    if kind == "gaussian":
        return add_gaussian(img, pick(spec.get("sigma", [5, 50])), rng)
    if kind == "speckle":
        return add_speckle(img, pick(spec.get("sigma", [10, 60])), rng)
    if kind == "speckle_gamma":
        return add_speckle_gamma(img, pick(spec.get("looks", [1.0, 10.0])), rng)
    if kind == "poisson":
        return add_poisson(img, pick(spec.get("peak", [5, 60])), rng)
    return add_salt_pepper(img, pick(spec.get("amount", [0.005, 0.05])), rng)
