# NST_for_Gen
* PyTorch implementation of Neural Style Transfer (NST) for image generation.
* The "assets" folder contains samples of input images to serve as an example to replicate the code. You need to replace the images with your own images if you want to use the proposed methodology.
* The "assets" folder also includes the saved models used for this algorithm. These files are not included in the docker image as it follows in the explanation. You need to manually replace those files locally in the same directory path as above if you choose to set up via our saved docker image.

## Setup
You can set up the algorithm in either of the following two ways:
1. Through Conda environment (Linux/macOS/Windows)
2. Through a Docker image (Linux/macOS)

* It is recommended that you use GPU capabilities to run the codes. [Nvidia/cuda](https://towardsdatascience.com/deep-learning-gpu-installation-on-ubuntu-18-4-9b12230a1d31) should be present to be able to use the GPU.

### 1/2) Setup through Conda environment
This project is prepared with:
- Ubuntu 18.04
- CUDA 10.0 (in case of using GPU)
- Conda

To install the rest of the dependencies, you can use the conda commands:
            
    conda env create --name=env-torch --file /NST_for_Gen/conda_env_app.yml
    conda activate env-torch
Note: In case Conda cannot install the dependencies with your system, you may try to remove version specifications in the "conda_env_app.yml" file, but that may cause inconsistencies in the code.

### 2/2) Setup through the Docker Image
You need to firstly request the docekr image from dockerhub, secondly build a docker container from the image, and finally run the full code.

#### Request the docker image using the command:
  >   $ sudo docker pull kenanmorani/nst_for_gen:latest
* The docker image includes full algorithm; generation part, and training and testing segmetnation part. Thus, pulling the image may take long depending.
     
 * To pull the docker image from dockerhub, make sure you have installed docker engine as in [here](https://docs.docker.com/engine/install/ubuntu/) for ubuntu system. <br/> <br/>
URL for the docker image is [https://hub.docker.com/r/kenanmorani/nst_for_gen/tags](https://hub.docker.com/r/kenanmorani/nst_for_gen/tags).      
#### Run the container:
* You need to download the "assets" folder to your local directory and "cd" your local directory to run the following commands from your local directory which contain the "assets" folder.
* To try the algorithm on your own images and masks, you need to replace the images samples in the folders with your own.

  >   $ docker run --gpus all -it -v "$PWD/assets:/NST_for_Gen/assets" kenanmorani/nst_for_gen:latest
  
  * If you want to try and run the code without gpu capabilities, you may detete "--gpus all" from the command above.
   
 * If you want to run docker as a non-root user, then you need to add your user to the docker group as in [here](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue).
 
 * In case of permission error for docker commands, run the command in the host terminal:
   >   $ newgrp docker
  
 * For giving a user full permissions to the generated files, run the following command in the host terminal:
   >   $ sudo chown -R $(id -u):$(id -g) assets/

## Running the codes
After installation, you are ready to run the full codes to generate data, as well as training and testing on the generated data uaing our algorithm.

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

## Acknowledgement
* This work is supported by the Scientific and Technological Research Council of Turkey (TUBITAK) under grant no 119E578.
* The data used in this study is collected under the Marie Curie IRG grant (no: FP7 PIRG08-GA-2010-27697).
## Citation
* If you find the work useful, please consider citing the paper:

@incollection{erdem2023automated, <br/>
  title={Automated analysis of phase-contrast optical microscopy time-lapse images: application to wound healing and cell motility assays of breast cancer}, <br/>
  author={Erdem, Yusuf Sait and Ayanzadeh, Aydin and Mayal{\i}, Berkay and Bal{\i}k{\c{c}}i, Muhammed and Belli, {\"O}zge Nur and U{\c{c}}ar, Mahmut and {\"O}zyusal, {\"O}zden Yal{\c{c}}{\i}n and Okvur, Devrim Pesen and {\"O}nal, Sevgi and Morani, Kenan and others}, <br/>
  booktitle={Diagnostic Biomedical Signal and Image Processing Applications with Deep Learning Methods}, <br/>
  pages={137--154}, <br/>
  year={2023}, <br/>
  publisher={Elsevier} <br/>
}
