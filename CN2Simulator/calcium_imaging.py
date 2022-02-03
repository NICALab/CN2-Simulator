import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from CN2Simulator.motif_gen import *
from CN2Simulator.utils.draw_util import create_NLS_neuron_2d, generate_centers

def create_calcium(spike_time, params, seed=0):
    """
    Generate calcium signal based on spike time and params.
    
    Arguments:
        spike_time: list of time of spikes for each neurons.
        params: dictionary of configuration
        seed: random seed
    
    Returns:
        calcium_signal: calcium signal created (numpy ndarray with shape [NID, bins])
    """
    # check arguments
    risetime = int(params["physiological"]["risetime"])   # milliseconds
    decaytime = int(params["physiological"]["decaytime"]) # milliseconds
    dF_F = float(params["physiological"]["dF_F"])         # ratio
    photobleaching = float(params["physiological"]["photobleaching"])
    baseline_low = float(params["physiological"]["baseline"][0])
    baseline_high = float(params["physiological"]["baseline"][1])
    recording_time = int(params["recording"]["recording_time"]) #seconds
    frame_rate = int(params["recording"]["frame_rate"])         # Hz
    
    if baseline_low < 0:
        raise Exception("baseline must be larger or equal to 0")
    if baseline_low > baseline_high:
        raise Exception("rhs of baseline must be larger or equal to lhs")
    
    # sampling
    rng = default_rng(seed)
    
    sample_arr = np.arange(1000/frame_rate, recording_time*1000, 1000/frame_rate).astype(int)
    sample_arr = sample_arr[sample_arr < recording_time * 1000]
    
    calcium_signal = []
    # draw single spike shape
    spikeshape = np.hstack((np.linspace(0, 1, num=int(risetime)*2+1),
                            np.exp(-np.arange(0.001, 0.001 * decaytime * 20, 0.001) * np.log(2) / (0.001 * decaytime))
                           )) * dF_F

    for nid in tqdm(range(len(spike_time)), desc="generating calcium signal", ncols=100):
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
        convolved_signal = np.convolve(bins, spikeshape, mode="full")[:recording_time * 1000]
        
        # add baseline and noise
        baseline = rng.uniform(baseline_low, baseline_high)
        baseline_photobleach = np.linspace(0, recording_time, recording_time*1000)
        baseline_photobleach = np.exp2(-baseline_photobleach / photobleaching)
        convolved_signal = baseline * baseline_photobleach * (convolved_signal + 1)
        
        # calcium signal must be larger than 0
        convolved_signal = np.maximum(convolved_signal, 0)
        
        # sampling
        convolved_signal = convolved_signal[sample_arr]

        calcium_signal.append(convolved_signal)
    
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
    if len(fov) not in [2]:
        raise Exception("FOV must be 2-dimensional")

    pixel_size = params["recording"]["pixel_size"]
    diameter_nucleus = params["shape"]["diameter_nucleus"] / pixel_size         # in pixels
    diameter_nucleus_std = params["shape"]["diameter_nucleus_std"] / pixel_size # in pixels
    distance = np.array(params["shape"]["distance"]) / pixel_size               # in pixels
    distance = distance.tolist()

    rng = default_rng(seed=seed)

    # generate centers
    centers = generate_centers(fov, min_dist=distance[0],\
        max_dist=distance[1], seed=seed)

    # generate calcium signal
    params["NIDs"] = centers.shape[0]
    spike_time, spike_time_motif = non_motif_gen(params, seed=seed)
    # (Type 1) Precise synchronous spikes
    gt1 = motif_gen(spike_time, spike_time_motif, 1, params, seed=1)
    # (Type 2) Precise sequential spikes
    gt2 = motif_gen(spike_time, spike_time_motif, 2, params, seed=2)
    # (Type 3) Precise temporal pattern
    gt3 = motif_gen(spike_time, spike_time_motif, 3, params, seed=3)
    # (Type 4) Rate-based synchronous pattern
    gt4 = motif_gen(spike_time, spike_time_motif, 4, params, seed=4)
    # (Type 5) Rate-based sequential pattern
    gt5 = motif_gen(spike_time, spike_time_motif, 5, params, seed=5)
    calcium_signal = create_calcium(spike_time, params, seed=seed+1)

    ground_truth = {
        "spike_time": spike_time,
        "spike_time_motif": spike_time_motif,
        "gt1": gt1,
        "gt2": gt2,
        "gt3": gt3,
        "gt4": gt4,
        "gt5": gt5,
        "calcium_signal": calcium_signal,
        "centers": centers
    }

    # generate calcium image
    calcium_image = np.zeros([calcium_signal.shape[1], *fov], dtype=np.float32)
    for idx, center in enumerate(tqdm(centers, desc="generating calicum image")):
        diameter_1 = rng.normal(diameter_nucleus, diameter_nucleus_std)
        diameter_2 = rng.normal(diameter_nucleus, diameter_nucleus_std)
        random_angle = rng.uniform(0, 360)
        # TODO: non-zero angle
        neuron = create_NLS_neuron_2d([diameter_1/2, diameter_2/2], angle=0, interpolation_ratio=4)
        calcium_image_neuron = np.outer(calcium_signal[idx], neuron)
        calcium_image_neuron = np.reshape(calcium_image_neuron, [-1] + list(neuron.shape))

        # shifts
        shifts = np.array(neuron.shape)/2
        shifts = shifts.astype(np.int32)
        calcium_image_start = center - shifts
        calcium_image_end = calcium_image_start + np.array(neuron.shape, dtype=np.int32)
        offset_start = np.maximum(calcium_image_start, 0) - calcium_image_start
        offset_end = calcium_image_end - np.minimum(calcium_image_end, fov)
        in_fov = np.array(neuron.shape, dtype=np.int32) > offset_start + offset_end # neuron inside fov
        if False in in_fov:
            continue
        
        # add signal
        calcium_image[:, calcium_image_start[0]+offset_start[0]:calcium_image_end[0]-offset_end[0],\
            calcium_image_start[1]+offset_start[1]:calcium_image_end[1]-offset_end[1]] +=\
             calcium_image_neuron[:, offset_start[0]:neuron.shape[0]-offset_end[0],\
             offset_start[1]:neuron.shape[1]-offset_end[1]]

    return calcium_image, ground_truth


if __name__=="__main__":
    pass