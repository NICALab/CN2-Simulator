import multiprocessing as mp

import numpy as np
from numpy.random import default_rng

from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter1d

from CN2Simulator.utils.util import load_params
from CN2Simulator.motif_gen import *

from scipy.io import loadmat

def create_voltage_helper(width, dF_F, recording_time,\
     spike_time, nid, baseline_low, baseline_high, photobleaching, add_noise, noise, sample_arr, simulation_rate, mode,\
        subthreshold_std, subthreshold_filter_std):

    # generate subthreshold voltage
    rng = default_rng(nid)
    current_width = rng.uniform(*width)
    current_df_f = rng.uniform(*dF_F)
    current_baseline = rng.uniform(baseline_low, baseline_high)

    white_noise = rng.normal(loc=0, scale=subthreshold_std, size=(simulation_rate * recording_time,))
    sub_threshold = gaussian_filter1d(white_noise, sigma=subthreshold_filter_std)

    # generate spikeshape
    num_frames = int(current_width / 2 * (simulation_rate / 1000))
    spikeshape = np.hstack((np.linspace(0, 1, num=num_frames+1),\
                            np.linspace(1, 0, num=num_frames+1))) * current_df_f
    spikeshape = np.hstack((spikeshape[:num_frames+1], spikeshape[num_frames+2:]))
    
    # TEST CODE
    # spikeshape = loadmat("hh_model_normalized.mat")["data_normalized"][0] * current_df_f
    # TEST CODE ENDS

    # change spike time to binning
    bins = np.zeros(recording_time * simulation_rate)
    for spiked in spike_time[nid]:
        if spiked < 0:
            continue
        idx = int(spiked // (1 / simulation_rate))
        ##SAMPLE
        # if idx % 20 != 0:
        #     idx = idx - (idx % 20)
        ##SAMPLE END
        if idx >= len(bins):
            continue
        bins[idx] += 1
        
    # convolve spikes
    convolved_signal = np.convolve(bins, spikeshape, mode="full")[:recording_time * simulation_rate]
    convolved_signal += sub_threshold
    
    # add baseline and noise
    if photobleaching == 0:
        baseline_photobleach = 1
    else:
        baseline_photobleach = np.linspace(0, recording_time, recording_time*simulation_rate)
        baseline_photobleach = np.exp2(-baseline_photobleach / photobleaching)
    convolved_signal = current_baseline * baseline_photobleach * (convolved_signal + 1)
    if add_noise:
        convolved_signal += noise
    
    # calcium signal must be larger than 0
    convolved_signal = np.maximum(convolved_signal, 0)

    # sampling
    if mode == "decimation":
        convolved_signal = convolved_signal[sample_arr]
    else:
        avg_signal = np.zeros_like(sample_arr, dtype=np.float64)
        for i in range(len(sample_arr)):
            if i == len(sample_arr) - 1:
                avg_signal[i] = np.mean(convolved_signal[sample_arr[i]:])
            else:
                avg_signal[i] = np.mean(convolved_signal[sample_arr[i]:sample_arr[i+1]])
        convolved_signal = avg_signal

    return convolved_signal, nid


def create_voltage(spike_time, params, add_noise=False, seed=0, verbose=True, simulation_rate=10000):
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
    width = [float(i) for i in params["physiological"]["width"]]          # milliseconds
    subthreshold_std = float(params["physiological"]["subthreshold_std"])
    subthreshold_filter_std = float(params["physiological"]["subthreshold_filter_std"])
    dF_F = [float(i) for i in params["physiological"]["dF_F"]]          # ratio
    photobleaching = float(params["physiological"]["photobleaching"])
    baseline_low = float(params["physiological"]["baseline"][0])
    baseline_high = float(params["physiological"]["baseline"][1])
    recording_time = int(params["recording"]["recording_time"])         # seconds
    frame_rate = int(params["recording"]["frame_rate"])                 # Hz
    mode = params["recording"]["mode"]                                  # binning / decimation
    if add_noise:
        noise = float(params["recording"]["noise"])                     # Standard deviation
    
    if baseline_low < 0:
        raise Exception("baseline must be larger or equal to 0")
    if baseline_low > baseline_high:
        raise Exception("rhs of baseline must be larger or equal to lhs")
    
    # sampling
    rng = default_rng(seed)
    
    sample_arr = np.arange(recording_time * frame_rate, dtype=np.int64)
    sample_arr = np.multiply(sample_arr, simulation_rate)
    sample_arr = np.divide(sample_arr, frame_rate).astype(np.int64)

    # import pdb; pdb.set_trace()
    
    # DEBUG START
    # nid = 0
    # current_width = rng.uniform(*width)
    # current_df_f = rng.uniform(*dF_F)
    # current_baseline = rng.uniform(baseline_low, baseline_high)
    # convolved_signal, nid = create_voltage_helper(current_width, current_df_f,\
    #         recording_time, spike_time, nid, current_baseline,\
    #         photobleaching, add_noise, 0, sample_arr, simulation_rate, mode,\
    #             subthreshold_std, subthreshold_filter_std)
    # savemat("test.mat", {"signal": convolved_signal})
    # DEBUG END

    
    voltage_signal = [[] for _ in range(len(spike_time))]
    pool = mp.Pool()

    def append_result(result):
        convolved_signal, nid = result
        voltage_signal[nid] = convolved_signal
        # savemat("./generated_data/DeepVID/calcium_signal_{}.mat".format(nid), {"calcium_signal": convolved_signal})
        if verbose:
            print(nid)

    for nid in tqdm(range(len(spike_time)), desc="generating calcium signal", ncols=100):
        pool.apply_async(create_voltage_helper, args=(width, dF_F,\
            recording_time, spike_time, nid, baseline_low, baseline_high,\
                photobleaching, add_noise, 0, sample_arr, simulation_rate, mode,\
                    subthreshold_std, subthreshold_filter_std), callback=append_result)
    
    pool.close()
    pool.join()

    voltage_signal = np.array(voltage_signal, dtype=np.float32)

    return voltage_signal



if __name__=="__main__":
    """
    Following VolPy implementation
    (1) sample spike times
    (2) convolving with a kernel matching the dynamics of Voltron signal in L1 neurons
    (3) sample sub-threshold activity by applying a Gaussian filter to white noise
    (4) exponential decaying to simulate photo-bleaching
    """
    # Load simulation parameters
    params = load_params("params.yaml")
    voltron_template = loadmat("Voltron_template_interp.mat")
    voltron_template_avg = voltron_template["template_avg_int"][0]
    voltron_template_avg -= voltron_template_avg.max()
    voltron_template_std = voltron_template["template_std_int"][0]

    # sample spike times
    spike_time, spike_time_motif = non_motif_gen(params, seed=0)

    # Convert to calcium imaging format
    voltage_signal = create_voltage(spike_time, params, seed=1)
    savemat("./generated_data/sampling_rate_matching.mat", {"voltage_signal": voltage_signal, "spike_time": spike_time})
    import pdb; pdb.set_trace()

    # sample sub-threshold activity
    rng = np.random.default_rng()
    white_noise = rng.normal(loc=0, scale=1.0, size=(2000,))
    sub_threshold = gaussian_filter1d(white_noise, sigma=2)


    import matplotlib.pyplot as plt
    plt.plot(sub_threshold)
    plt.show()


    # savemat("./generated_data/sub_thr.mat", {
    #     "white_noise": white_noise,
    #     "sub_threshold_10": gaussian_filter1d(white_noise, sigma=10),
    #     "sub_threshold_100": gaussian_filter1d(white_noise, sigma=100),
    #     "sub_threshold_1000": gaussian_filter1d(white_noise, sigma=1000)
    #     })