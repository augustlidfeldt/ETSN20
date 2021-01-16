# Automated Grad-CAM generation for testing of image recognition models in an MLOps context 

This repository contains source code and a replication package for a paper submitted to a workshop.

This code has been tested with **Python 3.7** and conda environments on MacOS and Win10.

## Getting started with the default example
1. Clone this repo.
1. Install the requirements.
```
pip install -r requirements-37.txt
```
1. Run ```run_example.py``` with the following arguments to specify the default file paths.
```
python run_example.py --model-path "model/Tunnel_model.h5" --weights-path "model/Tunnel_model_weights.h5" --dataset-path "images/day"
```
1. Examine output files in the subfolder "images/day/Grad-CAMs"
- Individual Grad-CAMs are created for each corresponding input file with the suffix -Grad-CAM
- A combined Grad-CAMs is created in ```_combined_Grad-CAMs.jpg```
- A combined image is created in ```_combined_images.jpg```

## Running Grad-CAM generation on other Keras models
1. Specify the locations of the model file, the weights file, and the input dataset using the command line arguments.
1. Specify the labels used by the model in ```run_example.py```.

## Authors

**August Lidfeldt** - [augustlidfeldt](https://github.com/augustlidfeldt)
**Simon Åberg** - [simanaberg](https://github.com/simanaberg)
**Ludwig Hedlund** - [luuddan](https://github.com/luuddan)
**Arvid Ekblom** - [arvidekblom](https://github.com/arvideklbom)
**Markus Borg** - [mrksbrg](https://github.com/mrksbrg)
