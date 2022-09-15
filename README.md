### Example testing command:
    $ python test.py --content "input/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t03_ch00.png" --style "input/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t03_ch00.tif" --style_mask "input/MCF7_SEMA6D_wound_healing_Mark_and_Find_001_LacZ_p002_t03_ch00.png"

### Example training command:
    $ python train.py --content_dir "/path/to/masks" --style_dir "/path/to/data"

### Example evaluation data generation command:
    $ python eval.py --content_dir "/path/to/masks" --style_dir "/path/to/data"
    
# Dependencies:
    - python=3.7.8
    - pillow=6.1
    - anaconda::cudatoolkit=10.0
    - pytorch=1.3.1
    - torchvision=0.4.2 
    - conda-forge::pytorch-lightning=1.0.8  
    - matplotlib=3.3.1
    - scikit-image=0.16.2
    - scikit-learn=0.23.1
    - salilab::py-opencv=3.4.2

# Pulling the docker image from dockerhub:
The docker image is saved on docker hub, and can be requested using the command:
     $ sudo docker pull kenanmorani/nst_for_gen:latest
