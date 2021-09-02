from bisect import bisect_left

def insert_spike(spike_times, spike_time_motif, nid, spiked, params):
    """
    Insert a spiked time to spike_times considering refractory period.
    
    Args:
        spike_times: list of time of spikes for each neurons.
        spike_time_motif: same as spike_time but only record motif-induced
        nid: neuron ID
        spiked: spiked time
        params: dictionary of configuration
    
    Returns:
        None
    """
    refractory_period = params["physiological"]["refractory_period"] / 1000;
    # delete refractory period
    left_idx = bisect_left(spike_times[nid], spiked - refractory_period)
    right_idx = bisect_left(spike_times[nid], spiked + refractory_period)
    if left_idx == right_idx: # nothing to delete
        pass
    else:
        del spike_times[nid][left_idx:right_idx]
    
    # delete refractory period (for spike_time_motif)
    left_idx_motif = bisect_left(spike_time_motif[nid], spiked - refractory_period)
    right_idx_motif = bisect_left(spike_time_motif[nid], spiked + refractory_period)
    if left_idx_motif == right_idx_motif: # nothing to delete
        pass
    else:
        del spike_time_motif[nid][left_idx_motif:right_idx_motif]
        
    # insert spiked (for spike_times)
    insertion_pt = bisect_left(spike_times[nid], spiked)
    spike_times[nid].insert(insertion_pt, spiked)
    
    # insert spiked (for spike_time_motif)
    insertion_pt = bisect_left(spike_time_motif[nid], spiked)
    spike_time_motif[nid].insert(insertion_pt, spiked)
    
    return None

    
if __name__=="__main__":
    # test 1
    spike_times = [[0.1, 0.4, 1.5]]
    insert_spike(spike_times, 0, 0.402, {"refractory_period": 5})
    print(spike_times == [[0.1, 0.402, 1.5]])
    # test 2
    spike_times = [[0.1, 0.4, 0.401, 1.5]]
    insert_spike(spike_times, 0, 0.401, {"refractory_period": 5})
    print(spike_times == [[0.1, 0.401, 1.5]])
    # test 3
    spike_times = [[0.1, 0.4, 0.401, 1.5]]
    insert_spike(spike_times, 0, 0.4005, {"refractory_period": 5})
    print(spike_times == [[0.1, 0.4005, 1.5]])
    