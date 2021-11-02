# Simul Motif

Comprehensive neuronal dynamics simulation tool including 

* electrophysiological recording
* calcium imaging
* voltage imaging
* motif simulation

---

## Features

* Camera settings
  * recording time / frame rate / exposure time
  * Field of View
  * pixel size
* Various microscopy simulation
  * SPIM
  * XLFM
* Neuron physiological parameters
  * refractory period
  * photobleaching
  * protein expression rate
  * GECI
    * rise/decay time
    * dF/F0
* Recording proteins
  * GEVI/GECI
  * H2B(NLS), cytosolic, soma targeted, membrane targeted
* Various motif types
  * synchronous firing
  * sequentially firing
  * arbitrary pattern of firing
  * synchronous firing rate increase
  * sequential firing rate increase

---

## How to use

- Install dependencies through `conda env create -f environment.yml`.

- Activate the conda environment through `conda activate SimulMotif`.

- Follow `example.ipynb` to simulate neuron motif.

- One can change `params.yaml` for different simulation settings.

- Check the data inside `\generated_data` folder.

- Visualize motifs with `SimulMotif/visualize.m`, which is a Matlab script.

---

## Figures
<p align="center">
    <img src="./docs/raster_plot.png" width="70%">
    <img src="./docs/calcium_signal.png" width="70%">
</p>