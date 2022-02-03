import numpy as np

from scipy.io import savemat

from CN2Simulator.utils.util import load_params
from CN2Simulator.motif_gen import *
from CN2Simulator.calcium_imaging import create_calcium, draw_calcium_image


if __name__=="__main__":
    # Load simulation parameters
    params = load_params("params.yaml")

    # Genereate non-motif activity
    # spike_time: list containing every spikes
    # spike_time_motif: list containing spikes induced by motifs
    spike_time, spike_time_motif = non_motif_gen(params, seed=0)

    # Generate motif activity
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

    # Convert to calcium imaging format
    if False:
        calcium_signal = create_calcium(spike_time, params, seed=6)

    # save spike time and motifs
    spike_time = np.array(spike_time, dtype=object)
    spike_time_motif = np.array(spike_time_motif, dtype=object)
    savemat("./generated_data/spike_time.mat", {"spike_time": spike_time,
                                                "spike_time_motif": spike_time_motif})

    savemat("./generated_data/gt_type_1.mat", {"gt_type_1": gt1})                                            
    savemat("./generated_data/gt_type_2.mat", {"gt_type_2": gt2})
    savemat("./generated_data/gt_type_3.mat", {"gt_type_3": gt3})
    savemat("./generated_data/gt_type_4.mat", {"gt_type_4": gt4})
    savemat("./generated_data/gt_type_5.mat", {"gt_type_5": gt5})
    
    if False:
        savemat("./generated_data/calcium_signal.mat", {"calcium_signal": calcium_signal})