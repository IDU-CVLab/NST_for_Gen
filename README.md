# NST_for_Gen
* PyTorch implementation of Neural Style Transfer (NST) for image generation method.
* The "assets" folder contains sample of input images to serve as an example to for replicating the code. You need to replace the input images in this folder with your own input images to be able to run the codes.
* The assets folder also includes the saved models used for this algorithm. This files are not included in the docker image as it follows in the explanation. You need to manually replace those files locally in the same directory path as above if you choose to set up via our saved docker image.

## Setup
You can setup in either of the following two ways:
1. Through Conda environment (Linux/macOS/Windows)
2. Through a Docker image (Linux/macOS)

* [Nvidia/cuda](https://towardsdatascience.com/deep-learning-gpu-installation-on-ubuntu-18-4-9b12230a1d31) should be present to be able to use the GPU for this algorithm.

### 1/2) Setup through Conda environment
This project is prepared upon:
- Ubuntu 18.04
- CUDA 10.0 (in case of using GPU)
- Conda

To install rest of the dependencies, you can use the conda commands:
            
    conda env create --name=env-torch --file /NST_for_Gen/conda_env_app.yml
    conda activate env-torch
Note: In case Conda cannot install the dependencies with your system, you may try to remove version specifications in the "conda_env_app.yml" file, but that may cause inconsistencies in the code.

### 2/2) Setup through the Docker Image
You need to firstly request the docekr image from dockerhub, secondly build a docker container from the image, and finally run the full code.

#### Request the docker image using the command:
  >   $ sudo docker pull kenanmorani/nst_for_gen:latest
* The docker image includes full algorithm (generating part and training and testing segmetnation part). Thus, pulling the image could take up to three hours.
     
 * To pull the docker image from dockerhub, make sure you have installed docker engine as in [here](https://docs.docker.com/engine/install/ubuntu/) for ubuntu system. 
 You also need to log in to our dockerhub account with the username ID {kenanmorani} and the password; *please send an email to request the password*.
 
     > $ sudo docker login <br>
       > username kenanmorani <br>
       > password <Tubitak119e578>
       
#### Run the container:
* You need to download the "assets" folder to your local directory and "cd" your local directory to run the following commands from your local directory which contain the "assets" folder.
* To try the algorithm on your own images and masks, you need to replace the images samples in the folders with your own.

  >   $ docker run --gpus all -it -v "$PWD/assets:/NST_for_Gen/assets" kenanmorani/nst_for_gen:latest
  
  * If you want to try and run the code without gpu capabilities, you may detete "--gpus all" from the command above.
   
 * If you want to run docker as non-root user then you need to add your user to the docker group as in [here](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue).
 
 * In case of permission error for docker commands, run the command in the host terminal:
   >   $ newgrp docker
  
 * For giving a user full permissions on the generated files, run the following command in the host terminal:
   >   $ sudo chown -R $(id -u):$(id -g) assets/

## Running the codes
After installations, you are ready to run the full codes for generating data,training, and testing on the generated data.

### Example testing command:
* This command runs on the sampled images in the 'assets' folder. Please change the directories in the code to reflect your own project directories.
```bash
$ python src/test.py --content_dir "assets/sample_input/masks/" --style_dir "assets/sample_input/data/" --style_mask_dir "assets/sample_input/masks/"
```
_For installation with Docker Image, run the container and the following command in it:_
```bash
# . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/test.py --content_dir "assets/sample_input/masks/" --style_dir "assets/sample_input/data/" --style_mask_dir "assets/sample_input/masks/"
```
### Example training command:
* Please change the directories in the code to reflect your own project directories.
```bash
$ python src/train.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
_For installation with Docker Image, run the container and the following command in it:_
```bash
# . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/train.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
### Example evaluation of data generation command:
* Please change the directories in the code to reflect your own project directories.
```bash
$ python src/eval.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
_For installation with Docker Image, run the container and the following command in it:_
```bash
# . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/eval.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
<br/> <br/>
* Please send an email in csae of issues or to request the duckerhub password.
## Citation
* If you find the work useful, please consider citing the paper:
> TODO: Get citation info.

## Acknowledgements
> TODO: 
