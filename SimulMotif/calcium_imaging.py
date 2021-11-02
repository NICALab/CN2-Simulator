import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

def create_calcium(spike_time, params, seed=None):
    """
    Generate calcium signal based on spike time and params.
    
    Arguments:
        spike_time: list of time of spikes for each neurons.
        params: dictionary of configuration
        seed: random seed
    
    Returns:
        calcium_signal
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
    noise = float(params["recording"]["noise"])                 # Standard deviation
    
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
        convolved_signal += rng.normal(0, noise, recording_time * 1000)
        
        # calcium signal must be larger than 0
        convolved_signal = np.maximum(convolved_signal, 0)
        
        # sampling
        convolved_signal = convolved_signal[sample_arr]
        
        calcium_signal.append(convolved_signal)
        
    return np.array(calcium_signal)


def draw_calcium_image(calcium_signal, params, dimension=2, type="NLS", seed=None):
    """
    Generate calcium imaging data based on calcium signal
    
    Arguments:
        calcium_signal: calcium signal trace from "create_calcium" function.
        dimension: whether the output is 2/3-dimensional
        type: one of ["NLS"]
            "NLS": nuclear-localized
        seed: random seed
    
    Returns:
        calcium_image: [T, d1, d2, [d3]]
    """
    #check arguments
    if not dimension in [2, 3]:
        raise Exception("dimension must be 2 or 3")

    # draw image
    