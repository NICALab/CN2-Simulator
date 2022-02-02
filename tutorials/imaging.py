import yaml
import numpy as np
import skimage.io as skio

from scipy.io import savemat
from SimulMotif.motif_gen import *
from SimulMotif.calcium_imaging import create_calcium, draw_calcium_image


if __name__=="__main__":
    # Load simulation parameters
    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Convert to calcium imaging format
    calcium_image, ground_truth = draw_calcium_image(params)

    # save spike time and motifs
    ground_truth["spike_time"] = np.array(ground_truth["spike_time"], dtype=object)
    ground_truth["spike_time_motif"] = np.array(ground_truth["spike_time_motif"], dtype=object)
    skio.imsave("./generated_data/test.tif", calcium_image)
    savemat("./generated_data/ground_truth.mat", ground_truth)
