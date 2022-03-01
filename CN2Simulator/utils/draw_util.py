import math
import numpy as np

from skimage.transform import resize
from scipy import ndimage
from scipy import sparse


def create_soma_neuron_2d():
    """
    Create soma-targeted calcium intensity

    Arguments:
        radius: the radius of ellipse in pixels
        angle: the ccw rotation angle in degrees (float)
        sigma: the gaussian sigma (float)

    Returns:
        neuron: results soma-targeted calcium intensity (numpy ndarray [np.float32])

        
    """
    pass


def create_NLS_neuron(fov, fov_pixel, center, radius, angle, sigma=1.4, simulation_unit=0.1, sparse_matrix=True):
    """
    Create nuclear localized calcium intensity

    Arguments:
        fov: in micrometers ([float], [y, x, [z]] order)
        center: in micrometers ([float], [y, x, [z]] order)
        radius: the radius of ellipse/ellipsoid in micrometers ([float], [y, x, [z]] order])
        pixel_size: pixel size of the ellipse/ellipsoid ([float], [y, x, [z]] order])
        angle: the ccw rotation angle in degrees (float)
        sigma: the gaussian sigma (float)
        simulation_unit: simulation unit in micrometers (float)

    Returns:
        neuron: resulting NLS calcium intensity (numpy ndarray [np.float32])
    """
    # check arguments
    if len(radius) not in [2, 3]:
        raise Exception("length of radius must be 2 or 3")
    if False in [type(i) in [float, int] for i in radius]:
        raise Exception("elements of radius must be a float or an integer")
    if not (type(sigma) in [float, int]):
        raise Exception("sigma must be a float or an integer")
    if not (type(angle) in [float, int]):
        raise Exception("angle must be a float or an integer")

    # (interpolation) increase the pixels temporarily
    # original_radius = radius
    # radius = [int(i * interpolation_ratio) for i in radius]
    
    # generate mask
    if len(radius) == 2:
        y, x = np.meshgrid(np.arange(0, fov[0], simulation_unit),\
            np.arange(0, fov[1], simulation_unit))
        dist_from_center = np.sqrt(((y-center[0])/radius[0])**2 + ((x-center[1])/radius[1])**2)
    else:
        y, x, z = np.meshgrid(np.arange(0, fov[0], simulation_unit),\
             np.arange(0, fov[1], simulation_unit),\
                 np.arange(0, fov[2], simulation_unit))
        dist_from_center = np.sqrt(((y-center[0])/radius[0])**2 + ((x-center[1])/radius[1])**2 +\
            ((z-center[2])/radius[2])**2)

    mask = dist_from_center <= 1

    # generate gaussian structure (NLS)
    gauss = np.exp(-dist_from_center**2 / (2.0 * sigma ** 2))
    neuron = gauss * mask
    neuron = neuron.astype(np.float32)

    # rotation
    # neuron = ndimage.rotate(neuron, angle=angle)

    # (interpolation) zoom
    neuron = resize(neuron, fov_pixel)

    # positive
    neuron = np.maximum(neuron, 0)

    if sparse_matrix:
        if len(radius) == 2:
            neuron = sparse.csr_matrix(neuron)
        else:
            neuron = [sparse.csr_matrix(neuron[z]) for z in range(neuron.shape[2])]

    return neuron


def random_perturb_l2(coords, max_perturbation, seed=0):
    """
    randomly perturbs the coordinates

    Arguments:
        coords: coordinates (np.float32 with size [N, 2] or [N, 3])
        max_perturbation: maximum amount of perturbation in L2 norm (float)
        seed: seed for random perturbation (int)
    
    Returns:
        coords: perturbed coordinates (numpy ndarray [np.float32])
    """
    # check arguments
    if not np.issubdtype(coords.dtype, np.floating):
        raise Exception("coords must be np.float32")
    if not(coords.shape[1] in [2, 3]):
        raise Exception("coordinates must be 2 or 3 dimension")
    if type(max_perturbation) not in [float, int]:
        raise Exception("max_perturbation must be a float or an integer")
    if type(seed) != int:
        raise Exception("seed must be an integer")

    rng = np.random.default_rng(seed=seed)

    if coords.shape[1] == 2:
        r = rng.uniform(0, max_perturbation, coords.shape[0])
        phi = rng.uniform(0, 2*np.pi, coords.shape[0])
        coords[:, 0] += r * np.cos(phi)
        coords[:, 1] += r * np.sin(phi)
    elif coords.shape[1] == 3:
        r = rng.uniform(0, max_perturbation, coords.shape[0])
        theta = rng.uniform(0, np.pi, coords.shape[0])
        phi = rng.uniform(0, 2*np.pi, coords.shape[0])
        coords[:, 0] += r * np.cos(phi) * np.sin(theta)
        coords[:, 1] += r * np.sin(phi) * np.sin(theta)
        coords[:, 2] += r * np.cos(theta)

    return coords


def generate_centers_fcc(shape, min_dist, max_dist, seed=0, discard_negatives=False):
    """
    Generate centers in face-centered cubic shape

    Arguments:
        shape: shape of the image or volume in micrometers ([float])
        min_dist: the minimum distance between centers in micrometers (float)
        max_dist: the maximum distance between centers in micrometers (float)
        seed: seed for random perturbation (int)
        discard_negatives: whether or not to discard negative centers (bool)
    
    Returns:
        coords: coordinates of the center (numpy ndarray [np.floating])
    """
    if not (len(shape) in [2, 3]):
        raise Exception("dimension of shape must be 2 or 3")
    if False in [type(i) in [float, int] for i in shape]:
        raise Exception("elements of shape must be an integer or float")
    if not (type(min_dist) in [float, int]):
        raise Exception("min_dist must be a float or an integer")
    if not (type(max_dist) in [float, int]):
        raise Exception("max_dist must be a float or an integer")
    if min_dist > max_dist:
        raise Exception("max_dist must be larger or equal to min_dist")
    
    # draw initial centers
    avg_dist = (min_dist + max_dist) / 2
    max_movement = (avg_dist - min_dist) / 2

    if len(shape) == 2:
        d1_idx = np.arange(0, shape[0], avg_dist * math.sqrt(2))
        d2_idx = np.arange(0, shape[1], avg_dist * math.sqrt(2))
        coords_1 = np.stack(np.meshgrid(d1_idx, d2_idx), -1).reshape(-1, 2)
        coords_2 = coords_1 + np.array([avg_dist/math.sqrt(2), avg_dist/math.sqrt(2)])
        coords_2 = np.array([i for i in coords_2 if (i[0] < shape[0] and i[1] < shape[1])])
        coords = np.concatenate((coords_1, coords_2), axis=0)
    else:
        d1_idx = np.arange(0, shape[0], avg_dist * math.sqrt(2))
        d2_idx = np.arange(0, shape[1], avg_dist * math.sqrt(2))
        d3_idx = np.arange(0, shape[2], avg_dist * math.sqrt(2))
        coords_1 = np.stack(np.meshgrid(d1_idx, d2_idx, d3_idx), -1).reshape(-1, 3)
        coords_2 = coords_1 + np.array([avg_dist/math.sqrt(2), avg_dist/math.sqrt(2), 0])
        coords_3 = coords_1 + np.array([avg_dist/math.sqrt(2), 0, avg_dist/math.sqrt(2)])
        coords_4 = coords_1 + np.array([0, avg_dist/math.sqrt(2), avg_dist/math.sqrt(2)])
        coords_2 = np.array([i for i in coords_2 if (i[0] < shape[0] and i[1] < shape[1] and i[2] < shape[2])])
        coords_3 = np.array([i for i in coords_3 if (i[0] < shape[0] and i[1] < shape[1] and i[2] < shape[2])])
        coords_4 = np.array([i for i in coords_4 if (i[0] < shape[0] and i[1] < shape[1] and i[2] < shape[2])])
        coords = np.concatenate((coords_1, coords_2, coords_3, coords_4), axis=0)

    # perturb points
    coords = random_perturb_l2(coords, max_movement, seed=seed)

    # exclude negative coords
    if discard_negatives:
        is_positive = coords >= 0
        is_positive_and = np.logical_and(is_positive[:, 0], is_positive[:, 1])
        if len(shape) == 3:
            is_positive_and = np.logical_and(is_positive_and, is_positive[:, 2])
        coords = coords[is_positive_and, :]

    return coords


def generate_centers(shape, min_dist, max_dist, seed=0, discard_negatives=False):
    """
    Generate centers according to the distance between neurons

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
    if False:
        import skimage.io as skio
        coords = generate_centers_fcc([100, 100, 100], min_dist=20, max_dist=20)
        image = np.zeros([100, 100, 100], dtype=np.float32)
        for i in coords:
            try:
                image[int(i[0]), int(i[1]), int(i[2])] = 1
            except:
                pass
        skio.imsave("generated_data/centers_3D.tif", image)
    if False:
        import skimage.io as skio
        coords = generate_centers_fcc([100, 100], min_dist=5, max_dist=20)
        image = np.zeros([100, 100], dtype=np.float32)
        for i in coords:
            try:
                image[int(i[0]), int(i[1])] = 1
            except:
                pass
        skio.imsave("generated_data/centers.tif", image)
    if False:
        print(random_perturb_l2(np.array([[0, 0], [1, 1]], dtype=np.float32), 1, seed=0))
        print(random_perturb_l2(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), 1, seed=0))
    if False:
        import skimage.io as skio
        coords = generate_centers([100, 100], min_dist=10, max_dist=20)
        image = np.zeros([100, 100], dtype=np.float32)
        for i in coords:
            image[i[0], i[1]] = 1
        skio.imsave("centers.tif", image)
    if True:
        import skimage.io as skio
        mask = create_NLS_neuron(fov=[100, 100], center=[30, 20], radius=[10, 10], angle=0, simulation_unit=0.1)
        skio.imsave("./generated_data/NLS_mask_0p1.tif", mask.toarray())
    if False:
        import skimage.io as skio
        gauss = gaussian_kernel(sigma=1)
        skio.imsave("gauss.tif", gauss.astype(np.float32))
        