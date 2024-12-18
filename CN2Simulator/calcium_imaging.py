import multiprocessing as mp

import numpy as np
from numpy.random import default_rng
from scipy import sparse, signal
from scipy.io import savemat
from tqdm import tqdm

from CN2Simulator.motif_gen import *
from CN2Simulator.utils.draw_util import create_NLS_neuron, generate_centers, generate_centers_fcc

def create_calcium_helper(current_rise_time, current_decay_time, current_df_f, recording_time,\
     spike_time, nid, baseline, photobleaching, add_noise, noise, sample_arr):
    # generate spikeshape
    spikeshape = np.hstack((np.linspace(0, 1, num=int(current_rise_time)*2+1),
                            np.exp(-np.arange(0.001, 0.001 * current_decay_time * 20, 0.001) * np.log(2) / (0.001 * current_decay_time))
                           )) * current_df_f

    # change spike time to binning
    bins = np.zeros(recording_time * 1000)
    for spiked in spike_time[nid]:
        if spiked < 0:
            continue
        idx = int(spiked // 0.001)
        if idx >= len(bins):
            continue
        bins[idx] += 1
    
    # convolve spikes
    # convolved_signal = np.convolve(bins, spikeshape, mode="full")[:recording_time * 1000]
    convolved_signal = signal.fftconvolve(bins, spikeshape, mode="full")[:recording_time * 1000]
    
    # add baseline and noise
    if photobleaching == 0:
        baseline_photobleach = 1
    else:
        baseline_photobleach = np.linspace(0, recording_time, recording_time*1000)
        baseline_photobleach = np.exp2(-baseline_photobleach / photobleaching)
    convolved_signal = baseline * baseline_photobleach * (convolved_signal + 1)
    if add_noise:
        convolved_signal += noise
    
    # calcium signal must be larger than 0
    convolved_signal = np.maximum(convolved_signal, 0)
    
    # sampling
    convolved_signal = convolved_signal[sample_arr]

    return convolved_signal, nid


def create_calcium(spike_time, params, add_noise=False, seed=0, verbose=True):
    """
    Generate calcium signal based on spike time and params.
    
    Arguments:
        spike_time: list of time of spikes for each neurons. ([[float]])
        params: dictionary of configuration
        add_noise: whether or not to add noise to the signal (bool)
        seed: random seed
    
    Returns:
        calcium_signal: calcium signal created (numpy ndarray with shape [NID, bins])
    """
    # check arguments
    risetime = [int(i) for i in params["physiological"]["risetime"]]   # milliseconds
    decaytime = [int(i) for i in params["physiological"]["decaytime"]] # milliseconds
    dF_F = [float(i) for i in params["physiological"]["dF_F"]]         # ratio
    photobleaching = float(params["physiological"]["photobleaching"])
    baseline_low = float(params["physiological"]["baseline"][0])
    baseline_high = float(params["physiological"]["baseline"][1])
    recording_time = int(params["recording"]["recording_time"]) #seconds
    frame_rate = int(params["recording"]["frame_rate"])         # Hz
    if add_noise:
        noise = float(params["recording"]["noise"])                 # Standard deviation
    
    if baseline_low < 0:
        raise Exception("baseline must be larger or equal to 0")
    if baseline_low > baseline_high:
        raise Exception("rhs of baseline must be larger or equal to lhs")
    
    # sampling
    rng = default_rng(seed)
    
    sample_arr = np.arange(recording_time * frame_rate, dtype=np.int64)
    sample_arr = np.multiply(sample_arr, 1000)
    sample_arr = np.divide(sample_arr, frame_rate).astype(np.int64)

    # sample_arr = np.arange(1000/frame_rate, recording_time*1000, 1000/frame_rate).astype(int)
    # sample_arr = sample_arr[sample_arr < recording_time * 1000]
    
    calcium_signal = [[] for _ in range(len(spike_time))]
    # draw single spike shape
    # spikeshape = np.hstack((np.linspace(0, 1, num=int(risetime)*2+1),
    #                         np.exp(-np.arange(0.001, 0.001 * decaytime * 20, 0.001) * np.log(2) / (0.001 * decaytime))
    #                        )) * dF_F

    pool = mp.Pool()

    def append_result(result):
        convolved_signal, nid = result
        calcium_signal[nid] = convolved_signal
        # savemat("./generated_data/DeepVID/calcium_signal_{}.mat".format(nid), {"calcium_signal": convolved_signal})
        if verbose:
            print(nid)

    for nid in tqdm(range(len(spike_time)), desc="generating calcium signal", ncols=100):
        current_rise_time = rng.uniform(risetime[0], risetime[1]+1)
        current_decay_time = rng.uniform(*decaytime)
        current_df_f = rng.uniform(*dF_F)
        pool.apply_async(create_calcium_helper, args=(current_rise_time, current_decay_time, current_df_f,\
            recording_time, spike_time, nid, rng.uniform(baseline_low, baseline_high),\
            photobleaching, add_noise, 0, sample_arr), callback=append_result)
    
    pool.close()
    pool.join()

    calcium_signal = np.array(calcium_signal, dtype=np.float32)

    return calcium_signal


def draw_calcium_image(params, seed=0):
    """
    Generate calcium imaging data based on calcium signal
    
    Arguments:
        calcium_signal: calcium signal trace from "create_calcium" function. 
        params: dictionary of configuration
        seed: random seed 
    
    Returns:
        calcium_image: (numpy ndarray with size [T, d1, d2, [d3]])
        ground_truth: dictionary including ground truth
    """
    # check arguments
    if not params["shape"]["type"] in ["NLS"]:
        raise Exception("type of shape is not supported")
    fov = params["recording"]["FOV"]
    if len(fov) not in [2, 3]:
        raise Exception("FOV must be 2/3-dimensional")

    simulation_unit = params["simulation"]["unit"]                  # micrometer
    pixel_size = params["recording"]["pixel_size"]                  # micrometer
    fov = [i * pixel_size for i in fov]                             # micrometer
    fov_simul_idx = [np.arange(0, i, simulation_unit) for i in fov]
    diameter_nucleus = params["shape"]["diameter_nucleus"]          # micrometer
    diameter_nucleus_std = params["shape"]["diameter_nucleus_std"]  # micrometer
    distance = np.array(params["shape"]["distance"])                # micrometer
    distance = distance.tolist()

    rng = default_rng(seed=seed)

    # generate centers
    centers = generate_centers_fcc(fov, min_dist=distance[0],\
        max_dist=distance[1], seed=seed)

    # generate calcium signal
    params["NIDs"] = centers.shape[0]
    spike_time, spike_time_motif = non_motif_gen(params, seed=seed)
    # (Type 1) Precise synchronous spikes
    # gt1 = motif_gen(spike_time, spike_time_motif, 1, params, seed=seed+1)
    # # (Type 2) Precise sequential spikes
    # gt2 = motif_gen(spike_time, spike_time_motif, 2, params, seed=seed+2)
    # # (Type 3) Precise temporal pattern
    # gt3 = motif_gen(spike_time, spike_time_motif, 3, params, seed=seed+3)
    # # (Type 4) Rate-based synchronous pattern
    # gt4 = motif_gen(spike_time, spike_time_motif, 4, params, seed=seed+4)
    # # (Type 5) Rate-based sequential pattern
    # gt5 = motif_gen(spike_time, spike_time_motif, 5, params, seed=seed+5)
    calcium_signal = create_calcium(spike_time, params, seed=seed+6)

    # generated neurons
    # TODO: non-zero angle
    gt_neuron_shapes = []
    for idx, center in enumerate(tqdm(centers, desc="generating neurons")):
        diameter_1 = rng.normal(diameter_nucleus, diameter_nucleus_std)
        diameter_2 = rng.normal(diameter_nucleus, diameter_nucleus_std)
        if len(fov) == 2:
            neuron = create_NLS_neuron(fov, params["recording"]["FOV"], center,\
                [diameter_1/2, diameter_2/2], angle=0, simulation_unit=simulation_unit,\
                sparse_matrix=True)
        else:
            diameter_3 = rng.normal(diameter_nucleus, diameter_nucleus_std)
            neuron = create_NLS_neuron(fov, params["recording"]["FOV"], center,\
                [diameter_1/2, diameter_2/2, diameter_3/2], angle=0, simulation_unit=simulation_unit,\
                sparse_matrix=True)
        gt_neuron_shapes.append(neuron)

    # generate calcium image
    calcium_image = np.zeros([calcium_signal.shape[1], *params["recording"]["FOV"]], dtype=np.float32)
    for idx, neuron in enumerate(tqdm(gt_neuron_shapes, desc="generating calcium image")):
        for t in range(calcium_signal.shape[1]):
            if len(fov) == 2:
                calcium_image[t] += calcium_signal[idx][t] * neuron
            else:
                for z, neuron_z in enumerate(neuron):
                    calcium_image[t, :, :, z] += calcium_signal[idx][t] * neuron_z

    # dictionary for save
    ground_truth = {
        "spike_time": spike_time,
        "spike_time_motif": spike_time_motif,
        "calcium_signal": calcium_signal,
        "gt_neuron_shapes": gt_neuron_shapes
    }

    return calcium_image, ground_truth


if __name__=="__main__":
    create_calcium()