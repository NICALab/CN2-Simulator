import numpy as np

from scipy import ndimage


def create_NLS_neuron_2d(radius, angle, sigma=1.4):
    """
    Create nuclear localized calcium intensity

    Arguments:
        radius: the radius of ellipse in pixels ([int])
        angle: the ccw rotation angle in degrees (float)
        sigma: the gaussian sigma (float)

    Returns:
        neuron: resulting NLS calcium intensity (numpy ndarray [np.float32])
    """
    # check arguments
    if len(radius) != 2:
        raise Exception("length of radius must be 2")
    if False in [type(i) == int for i in radius]:
        raise Exception("elements of radius must be an integer")
    if not (type(sigma) in [float, int]):
        raise Exception("sigma must be a float or an integer")
    if not (type(angle) in [float, int]):
        raise Exception("angle must be a float or an integer")
    
    # mask
    x, y = np.meshgrid(np.arange(-radius[0], radius[0]+1), np.arange(-radius[1], radius[1]+1))
    dist_from_center = np.sqrt((x/radius[0])**2 + (y/radius[1])**2)
    mask = dist_from_center <= 1

    # structure
    gauss = np.exp(-dist_from_center**2 / (2.0 * sigma ** 2))
    neuron = gauss * mask
    neuron = neuron.astype(np.float32)

    # rotation
    neuron = ndimage.rotate(neuron, angle=angle)

    return neuron


def generate_centers(shape, min_dist, max_dist, seed=0, discard_negatives=False):
    """
    Generate centers according to the density uniformly

    Arguments:
        shape: shape of the image or volume ([int])
        min_dist: the minimum distance between centers in pixels (float)
        max_dist: the maximum distance between centers in pixels (float)
        discard_negatives: whether or not to discard negative centers (bool)
    
    Returns:
        coords: coordinates of the center (numpy ndarray [np.int32])
    """
    # check arguments
    if len(shape) != 2:
        raise Exception("length of shape must be 2")
    if False in [type(i) == int for i in shape]:
        raise Exception("elements of shape must be an integer")
    if not (type(min_dist) in [float, int]):
        raise Exception("min_dist must be a float or an integer")
    if not (type(max_dist) in [float, int]):
        raise Exception("max_dist must be a float or an integer")
    if min_dist > max_dist:
        raise Exception("max_dist must be larger or equal to min_dist")
    
    # draw initial centers
    avg_dist = (min_dist + max_dist) / 2
    max_movement = (avg_dist - min_dist) / 2

    d1_idx = np.arange(0, shape[1]-1, avg_dist)
    d2_idx = np.arange(0, shape[0]-1, avg_dist)

    # perturb points
    rng = np.random.default_rng(seed=seed)
    coords = np.stack(np.meshgrid(d1_idx, d2_idx), -1).reshape(-1, 2)
    noise = rng.uniform(-max_movement, max_movement, (len(coords), 2))
    coords += noise

    # exclude negative coords
    coords = np.around(coords)
    coords = coords.astype(np.int32)
    if discard_negatives:
        is_positive = coords >= 0
        is_positive_and = np.logical_and(is_positive[:, 0], is_positive[:, 1])
        coords = coords[is_positive_and, :]

    return coords


def gaussian_kernel(sigma):
    x, y = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-2, 2, 200))
    dst = np.sqrt(x * x + y * y)

    gauss = np.exp(- dst**2 / (2.0 * sigma ** 2))

    return gauss


if __name__=="__main__":
    if True:
        import skimage.io as skio
        coords = generate_centers([100, 100], min_dist=10, max_dist=20)
        image = np.zeros([100, 100], dtype=np.float32)
        for i in coords:
            image[i[0], i[1]] = 1
        skio.imsave("centers.tif", image)
    if False:
        import skimage.io as skio
        mask = create_NLS_neuron_2d(radius=[40, 30], angle=0)
        skio.imsave("NLS_mask.tif", mask)
    if False:
        import skimage.io as skio
        gauss = gaussian_kernel(sigma=1)
        skio.imsave("gauss.tif", gauss.astype(np.float32))
        