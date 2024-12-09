import numpy as np
import skimage.io as skio

from scipy.io import savemat

from CN2Simulator.utils.util import load_params
from CN2Simulator.motif_gen import *
from CN2Simulator.calcium_imaging import create_calcium, draw_calcium_image
from CN2Simulator.microscopy import record


if __name__=="__main__":
    # Load simulation parameters
    params = load_params("params.yaml")

    # Convert to calcium imaging format
    calcium_image, ground_truth = draw_calcium_image(params)
    # calcium_image = record(calcium_image, params, seed=0)

    # save spike time and motifs
    ground_truth["spike_time"] = np.array(ground_truth["spike_time"], dtype=object)
    ground_truth["spike_time_motif"] = np.array(ground_truth["spike_time_motif"], dtype=object)
    skio.imsave("./generated_data/test.tif", calcium_image)
    savemat("./generated_data/ground_truth.mat", ground_truth)
