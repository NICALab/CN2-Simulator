## Simul Motif

Neuronal motif simulation tool including electrical and calcium recording

<p align="center">
    <img src="./docs/raster_plot.png" width="100%">
</p>

<p align="center">
    <img src="./docs/calcium_signal.png" width="100%">
</p>
    

## How to use

- Install dependencies through `conda create --name <env> --file conda_req.txt`, fill `<env>` for your own environment name.

- Follow `example.ipynb` to simulate neuron motif.

- One can change `params.yaml` for different simulation settings.

- Check the data inside `\generated_data` folder.

- Visualize motifs with `SimulMotif/visualize.m`, which is a Matlab script.