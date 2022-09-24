ARG UBUNTU_VERSION=18.04
ARG CUDA_VERSION=11.4.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
COPY ./NST_for_Gen/ ./NST_for_Gen/
# update apt and get miniconda
RUN apt-get update \
    && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /Miniconda3

RUN /Miniconda3/bin/conda env create --name=env-torch --file /NST_for_Gen/conda_env_app.yml

RUN /Miniconda3/bin/conda init && bash ~/.bashrc && . ~/.bashrc

ENV conda /Miniconda3/bin/conda
ENV bashrc /root/.bashrc

# Run the test code:
#RUN . $bashrc  >/dev/null && conda activate env-torch  >/dev/null && cd /NST_for_Gen && python src/test.py --content_dir "assets/sample_input/masks/" --style_dir "assets/sample_input/data/" --style_mask_dir "assets/sample_input/masks/"
