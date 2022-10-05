# NST_for_Gen
PyTorch implementation of Neural Style Transfer (NST) for image generation method.

## Setup
Installation can be done in two ways:
1. Through Conda environment (Linux/macOS/Windows)
2. Through a Docker image (Linux/macOS)

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
* Request the docker image using the command:
  >   $ sudo docker pull kenanmorani/nst_for_gen:latest
  Pulling the image completly should take aproximately three hours.
     
 * To pull the docker image from dockerhub, make sure you have installed docker engine as in [here](https://docs.docker.com/engine/install/ubuntu/) for ubuntu system. 
 You also need to log in to our dockerhub account with the username ID {kenanmorani} and the password; *please send an email to request the password*.
 
     > $ sudo docker login <br>
       > username kenanmorani <br>
       > password <Tubitak119e578>
 * To run the container:
  >   $ docker run --gpus all -it -v "$PWD/assets:/NST_for_Gen/assets" nst_for_gen:latest
 
 * Nvidia/cuda should be present to be able to use the GPU for this algorithm.
    
 * If you want to run docker as non-root user then you need to add your user to the docker group as in [here](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue).
 
 * In case of permission error for docker commands, run the command in the host terminal:
   >   $ newgrp docker
 * For making any user to have full permissions on the generated files, run the command in the host terminal:
   >   $ sudo chown -R $(id -u):$(id -g) assets/
## Running
### Example testing command:
```bash
$ python src/test.py --content_dir "assets/sample_input/masks/" --style_dir "assets/sample_input/data/" --style_mask_dir "assets/sample_input/masks/"
```
For Docker container:
```bash
$ . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/test.py --content_dir "assets/sample_input/masks/" --style_dir "assets/sample_input/data/" --style_mask_dir "assets/sample_input/masks/"
```
### Example training command:
- Copy dataset to the "assets/dataset" directory, then run
```bash
$ python src/train.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
For Docker container:
```bash
$ . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/train.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
### Example evaluation data generation command:
- Copy dataset to the "assets/dataset" directory, then run
```bash
$ python src/eval.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
For Docker container:
```bash
$ . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/eval.py --content_dir "assets/dataset/masks" --style_dir "assets/dataset/data"
```
Please send me an email in csae of issues (kenan.morani@gmail.com).

##Citation
* If you find the work useful, please consider citing our paper:
> TODO: Get citation info.

##Acknowledgements
> TODO: 
