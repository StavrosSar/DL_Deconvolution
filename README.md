# Deep Learning Deconvolution Methods

This repository contains the code and resources for my thesis project.

## Overview

[The goal of this Thesis is to investigate and develop DL-based approaches for image deconvolution that are both computationally efficient and capable of producing high resolution and low noise image reconstructions from degraded images. The ultimate goal is to make a practical tool that can be applied in many fields like microscopy, astrophysics and medical imaging to reveal a clearer visualisation]


## Environment & Dependencies

The Anaconda environment used for this project can be downloaded and installed from the link below. This ensures all dependencies and correct versions are installed.

**Download Environment:** [Anaconda Envieroment](https://mega.nz/folder/VgkhzC5R#BCabPqdQoR5U8JGf-T-9vg)

## Datasets

The datasets required to train and/or evaluate the models are available at the following link:

**Download Datasets:** [Datasets](https://mega.nz/folder/5hVSBQ7B#-cOb0xHCxL02CAQ--G95Qg)

## Pre-trained Models

A pre-trained checkpoint for the SUNet model is available for download.

**Download SUNet Checkpoint:** [Pretrained path](https://mega.nz/folder/t5VEBDbR#LfVZjKI4emF9my-VziAetQ)

## Usage

[The steps in order to run the project:]
1.  Create and activate the conda environment from the provided file.
    ```bash
    conda env create -f environment.yml
    conda activate my_thesis_env
    ```
2.  Download the datasets and place them in the `data/` folder.
3.  Download the pre-trained checkpoint and update the path in the configuration file of the SUNet-main.
4.  To train the model, run:
    ```bash
    python train.py
    ```
5.  To evaluate the model, run:
    ```bash
    python evaluate.py
    ```


## Project Structure

[ Briefly explain the structure of your project folders.
For example:]


## Acknowledgments

- I would like to thank [Utsav Akhaury] for their [SUNet project], whose code for [the SUNet model] was adapted and used in this work.
  - Link to their repository: [https://github.com/utsav-akhaury/SUNet]



Datasets: https://mega.nz/folder/5hVSBQ7B#-cOb0xHCxL02CAQ--G95Qg
SUNet checkpoing path: https://mega.nz/folder/t5VEBDbR#LfVZjKI4emF9my-VziAetQ
Enviroment used in Anaconda: https://mega.nz/folder/VgkhzC5R#BCabPqdQoR5U8JGf-T-9vg