import numpy as np


def record(calcium_image, params, seed=0):
    """
    Post-process simulating real-world microscopy

    Arguments:
        calcium_image: raw calcium image (numpy ndarray with size [T, d1, d2, [d3]])
        params: dictionary of configuration

    Returns:
        calcium_image: post-processed image (numpy ndarray with size [T, d1, d2, [d3]])
    """
    # check arguments
    microscopy = params["recording"]["microscopy"]
    if microscopy == "raw":
        return calcium_image
    gaussian_noise_std = params["recording"]["gaussian_noise"]
    dims = calcium_image.shape

    # add Gaussian noise
    rng = np.random.default_rng(seed=seed)
    gaussian_noise = rng.normal(0, gaussian_noise_std, dims)
    calcium_image += gaussian_noise

    return calcium_image