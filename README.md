# ETSN20
Project in Software Testing – ETSN20 Lund University

This repository contains code for a project in the course ETSN20 during the Fall semester of 2020.

## Setup and generation of GradCAMs
1. After downloading the repo and installing the requirements, open the [example file](https://github.com/augustlidfeldt/ETSN20/blob/main/run_example.py).

2. In the run_example, set the correct file paths to the images and model in the downloaded project folder.

3. If you want to run our tunnel model, leave the code as is. If you prefer running a pre-installed Keras model uncomment the last segment, and comment out the loading of the custom model.

4. Run the run_example.py file. GradCAMs should be generated and put in a folder called GradCAMs in the source image folder.

5. If you wish to run your own custom model, just change the image and model filepaths to the place were they're kept. Add your custom labels instead of the current ones.  Note that the weight file should be named as the model with the addition of "_weights.h5".


## Authors

**August Lidfeldt** - [augustlidfeldt](https://github.com/augustlidfeldt)
**Simon Åberg** - [simanaberg](https://github.com/simanaberg)
**Ludwig Hedlund** - [luuddan](https://github.com/luuddan)
**Arvid Ekblom** - [arvidekblom](https://github.com/arvideklbom)

## Acknowledgments

* Thanks to Markus Borg for supervising this project.

