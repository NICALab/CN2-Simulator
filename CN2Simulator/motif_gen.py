import math
import numpy as np

from tqdm import tqdm
from numpy.random import default_rng
from CN2Simulator.utils.util import insert_spike

def non_motif_gen(params, seed=None, verbose=True):
    """
    Generate non-motif-based spikes.
    When generating spikes, it considers the refractory period.
    
    Args:
        params: dictionary of configuration
        seed: random seed
    
    Returns:
        spike_time: list of time of spikes for each neurons.
        spike_time_motif: same as spike_time but only record motif-induced
    """
    
    # initialize
    rng = default_rng(seed)
    NIDs = int(params["NIDs"])
    simulation_time = int(params["recording"]["recording_time"]) # seconds
    refractory_period = params["physiological"]["refractory_period"] # milliseconds
    peak_to_mean = params["background"]["peak_to_mean"] # ratio
    firing_rate = params["background"]["firing_rate"] # Hz
    oscillation_frequency = params["background"]["oscillation_frequency"] # Hz
    coherent = params["background"]["coherent"] # Boolean
    # burst = np.array(params["background"]["burst"]) # probability
    # burst = burst / np.sum(burst)
    burst_lambda = params["background"]["burst_lambda"]
    intra_burst_time = params["background"]["intra_burst_time"] # milliseconds
    frame_rate = float(params["recording"]["frame_rate"]) # Hz
    spike_time = [[] for x in range(NIDs)]        # initialize entire spike time list
    spike_time_motif = [[] for x in range(NIDs)]  # initialize motif-induced spike time list
    
    # draw spike times
    NIDs_range = tqdm(range(NIDs), desc="Generating non-motif spikes") if verbose else range(NIDs)

    if peak_to_mean == 0: # stationary
        for NID in NIDs_range:
            non_motif_rate = rng.uniform(firing_rate[0], firing_rate[1])
            spiked = 0
            while True:
                burst_time = 1 + rng.poisson(lam=burst_lambda)
                spiked += rng.exponential(1/non_motif_rate) # + refractory_period / 1000
                for i in range(burst_time):
                    if spiked <= simulation_time:
                        spike_time[NID].append(spiked)
                    else:
                        break
                    if i == burst_time - 1:
                        spiked += (refractory_period / 1000)
                    else:
                        spiked += (rng.uniform(*intra_burst_time) / 1000)
                if spiked > simulation_time:
                    break
    else: # nonstationary
        # draw phase
        if coherent:
            phases = [0 for _ in range(NIDs)]
        else:
            phases = [rng.uniform(0, 2*math.pi) for _ in range(NIDs)]
        # draw spikes
        t_array = np.linspace(1/1000, simulation_time, num=simulation_time*1000)
        rad_array = (2 * math.pi * oscillation_frequency) * t_array
        
        for NID in range(NIDs):
            non_motif_rate = rng.uniform(firing_rate[0], firing_rate[1])
            instant_rate = np.maximum((peak_to_mean * np.sin(rad_array + phases[NID]) + 1) * non_motif_rate, 0)
            spiked = np.nonzero(rng.binomial(1, instant_rate / 1000))[0]
            spike_time[NID] = list(t_array[spiked])

    return spike_time, spike_time_motif

def motif_gen(spike_time, spike_time_motif, motif_type, params, seed=None):
    """
    Generate motif-based spikes and insert to spike_time.
    When inserting spikes, it considers the refractory period.
    If the previous spikes are within the refractory period, remove conflicting spikes.
    
    Args:
        spike_time: list of time of spikes for each neurons.
        spike_time_motif: same as spike_time but only record motif-induced
        motif_type: type of motifs from 1 to 5.
        params: dictionary of configuration
        seed: random seed
        
    Returns:
        ground_truth: list of dictionary containing ...
            ** spike time is lags + motif_times (if lags are returned)
            *** each element contains individual motifs
            (Common)
                NIDs: numpy array of neuron IDs, zero_indexed [int]
                motif_times: list of time of motif occurrence [float]
            (Type 2, 5)
                lags: list of time lag between spikes [float]
            (Type 3)
                lags: (double) list of time lag between spikes [float]
    """
    rng = default_rng(seed)
    NIDs = params["NIDs"]
    simulation_time = params["recording"]["recording_time"]
    refractory_period = params["physiological"]["refractory_period"]
    probabilistic_participation = params["noise"]["probabilistic_participation"]
    temporal_jitter = params["noise"]["temporal_jitter"] / 1000
    time_warping = params["noise"]["time_warping"] / 100
    ground_truth = []
    if motif_type == 1: # Precise synchronous spikes
        firing_rate = params["motif_type_1"]["firing_rate"]
        neurons = params["motif_type_1"]["neurons"]
        motifs = params["motif_type_1"]["motifs"]
        for motif in range(motifs):
            # choose neurons
            tmp_neurons = rng.choice(NIDs, neurons, replace=False)
            ground_truth.append({"NIDs": tmp_neurons})
            # insert spikes
            tmp_spikes = []
            spiked = rng.exponential(1/firing_rate)
            while spiked <= simulation_time:
                for nid in tmp_neurons:
                    if rng.binomial(1, probabilistic_participation) == 1:
                        jitter = rng.normal(0, temporal_jitter)
                        insert_spike(spike_time, spike_time_motif, nid, spiked + jitter, params)
                tmp_spikes.append(spiked)
                spiked += rng.exponential(1/firing_rate) + refractory_period / 1000
            ground_truth[-1]["motif_times"] = tmp_spikes
        
    elif motif_type == 2: # Precise sequential spikes
        firing_rate = params["motif_type_2"]["firing_rate"]
        neurons = params["motif_type_2"]["neurons"]
        motifs = params["motif_type_2"]["motifs"]
        min_lags = params["motif_type_2"]["min_lags"] / 1000
        max_lags = params["motif_type_2"]["max_lags"] / 1000
        for motif in range(motifs):
            # choose neurons
            tmp_neurons = rng.choice(NIDs, neurons, replace=False)
            ground_truth.append({"NIDs": tmp_neurons})
            # choose time lags
            tmp_lags = [0]
            for lag in rng.uniform(min_lags, max_lags, neurons-1):
                tmp_lags.append(tmp_lags[-1] + lag)
            ground_truth[-1]["lags"] = tmp_lags
            # insert spikes
            tmp_spikes = []
            spiked = rng.exponential(1/firing_rate)
            while spiked <= simulation_time:
                if time_warping == 0:
                    warp = 1
                else:
                    warp = rng.uniform(1, 1+time_warping)
                for idx, nid in enumerate(tmp_neurons):
                    if rng.binomial(1, probabilistic_participation) == 1:
                        jitter = rng.normal(0, temporal_jitter)
                        insert_spike(spike_time, spike_time_motif, nid, spiked + warp * tmp_lags[idx] + jitter, params)
                tmp_spikes.append(spiked)
                spiked += rng.exponential(1/firing_rate) + refractory_period / 1000
            ground_truth[-1]["motif_times"] = tmp_spikes

    elif motif_type == 3: # Precise temporal pattern
        firing_rate = params["motif_type_3"]["firing_rate"]
        neurons = params["motif_type_3"]["neurons"]
        motifs = params["motif_type_3"]["motifs"]    
        min_lags = params["motif_type_3"]["min_lags"] / 1000
        max_lags = params["motif_type_3"]["max_lags"] / 1000
        max_spikes = int(params["motif_type_3"]["max_spikes"])
        
        for motif in range(motifs):
            # choose neurons
            tmp_neurons = rng.choice(NIDs, neurons, replace=False)
            ground_truth.append({"NIDs": tmp_neurons})
            
            # choose neuron firing
            num_fire = []
            fire_sequence = [] # later filled with [NIDs]
            for neuron in range(neurons):
                num_fire.append(int(rng.uniform(1, max_spikes + 1)))
            for _ in range(sum(num_fire)):
                curr_nid = np.nonzero(num_fire)[0]
                idx = rng.choice(curr_nid, 1)[0]
                fire_sequence.append(tmp_neurons[idx])
                num_fire[idx] -= 1
                
            # choose time lags
            tmp_lags_flatten = [0]
            for lag in rng.uniform(min_lags, max_lags, len(fire_sequence)-1):
                tmp_lags_flatten.append(tmp_lags_flatten[-1] + lag)
            
            tmp_lags = [[] for _ in range(neurons)]
            for idx, NID in enumerate(fire_sequence):
                tmp_lags[np.argwhere(tmp_neurons == NID)[0][0]].append(tmp_lags_flatten[idx])
            ground_truth[-1]["lags"] = tmp_lags
                        
            # insert spikes
            tmp_spikes = []
            spiked = rng.exponential(1/firing_rate)
            while spiked <= simulation_time:
                if time_warping == 0:
                    warp = 1
                else:
                    warp = rng.uniform(1, 1+time_warping)
                for idx, nid in enumerate(tmp_neurons):
                    for window_idx in range(len(tmp_lags[idx])):
                        if rng.binomial(1, probabilistic_participation) == 1:
                            jitter = rng.normal(0, temporal_jitter)
                            insert_spike(spike_time, spike_time_motif, nid, spiked + warp * tmp_lags[idx][window_idx] + jitter, params)
                tmp_spikes.append(spiked)
                spiked += rng.exponential(1/firing_rate) + refractory_period / 1000
            ground_truth[-1]["motif_times"] = tmp_spikes

    elif motif_type == 4: # Rate-based synchronous pattern
        firing_rate = params["motif_type_4"]["firing_rate"]
        neurons = params["motif_type_4"]["neurons"]
        motifs = params["motif_type_4"]["motifs"]
        window_size = params["motif_type_4"]["window_size"]
        window_rate = params["motif_type_4"]["window_rate"]
        for motif in range(motifs):
            # choose neurons
            tmp_neurons = rng.choice(NIDs, neurons, replace=False)
            ground_truth.append({"NIDs": tmp_neurons})
            # insert spikes
            tmp_spikes = []
            spiked = rng.exponential(1/firing_rate)
            while spiked <= simulation_time:
                for idx, nid in enumerate(tmp_neurons):
                    window_lag = rng.exponential(1/window_rate)
                    while window_lag <= window_size:
                        if rng.binomial(1, probabilistic_participation) == 1:
                            jitter = rng.normal(0, temporal_jitter)
                            insert_spike(spike_time, spike_time_motif, nid, spiked + window_lag + jitter, params)
                        window_lag += rng.exponential(1/window_rate) + refractory_period / 1000
                tmp_spikes.append(spiked)
                spiked += rng.exponential(1/firing_rate) + refractory_period / 1000
            ground_truth[-1]["motif_times"] = tmp_spikes

    elif motif_type == 5: # Rate-based sequential pattern
        firing_rate = params["motif_type_5"]["firing_rate"]
        neurons = params["motif_type_5"]["neurons"]
        motifs = params["motif_type_5"]["motifs"]
        window_size = params["motif_type_5"]["window_size"]
        window_rate = params["motif_type_5"]["window_rate"]
        min_lags = params["motif_type_5"]["min_lags"] / 1000
        max_lags = params["motif_type_5"]["max_lags"] / 1000
        for motif in range(motifs):
            # choose neurons
            tmp_neurons = rng.choice(NIDs, neurons, replace=False)
            ground_truth.append({"NIDs": tmp_neurons})
            # choose time lags
            tmp_lags = [0]
            for lag in rng.uniform(min_lags, max_lags, neurons-1):
                tmp_lags.append(tmp_lags[-1] + lag)
            ground_truth[-1]["lags"] = tmp_lags
            # insert spikes
            tmp_spikes = []
            spiked = rng.exponential(1/firing_rate)
            while spiked <= simulation_time:
                if time_warping == 0:
                    warp = 1
                else:
                    warp = rng.uniform(1, 1+time_warping)
                for idx, nid in enumerate(tmp_neurons):
                    window_lag = rng.exponential(1/window_rate)
                    while window_lag <= window_size:
                        if rng.binomial(1, probabilistic_participation) == 1:
                            jitter = rng.normal(0, temporal_jitter)
                            insert_spike(spike_time, spike_time_motif, nid, spiked + warp * tmp_lags[idx] + window_lag + jitter, params)
                        window_lag += rng.exponential(1/window_rate) + refractory_period / 1000
                tmp_spikes.append(spiked)
                spiked += rng.exponential(1/firing_rate) + refractory_period / 1000
            ground_truth[-1]["motif_times"] = tmp_spikes

    else:
        raise Exception("not supported motif type")
        return False
    
    # check results
    for idx_1, spike_time_nid in enumerate(spike_time):
        last_index = None
        for idx, spike_time_nid_t in enumerate(reversed(spike_time_nid)):
            if spike_time_nid_t < simulation_time:
                last_index = idx
                break
        if (last_index is None) or (last_index == 0):
            continue
        spike_time[idx_1] = spike_time_nid[:-last_index]

    for idx_1, spike_time_nid in enumerate(spike_time_motif):
        last_index = None
        for idx, spike_time_nid_t in enumerate(reversed(spike_time_nid)):
            if spike_time_nid_t < simulation_time:
                last_index = idx
                break
        if (last_index is None) or (last_index == 0):
            continue
        spike_time_motif[idx_1] = spike_time_nid[:-last_index]
    
    return ground_truth
    
if __name__=="__main__":
    # (type 1) motif test
    pass