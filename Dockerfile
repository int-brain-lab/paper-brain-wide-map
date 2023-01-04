FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# This can optionally be built with just ubuntu, rather than the nvidia cuda container.
# If saving space is a concern, this is the way to go.
LABEL description="Core container which has the basic necessities to run analyses in the\
 IBL infrastructure. Does not contain packages involved in running IBL GUI apps, such as pyqt etc,\
 but can connect to the IBL database. Note that for IBL internal users you will need to mount your\
 credentials file to /root/.one, using e.g. the -v flag in docker run. Also, for the encoding\
 analyses you will need to mount a modified copy of the brainwidemap.encoding.params.py file which\
 has the appropriately modified paths to the data."
LABEL maintainer="Berk Gercek (@github:berkgercek)"
LABEL version="0.1"

WORKDIR /data
COPY ./environment.yaml /data/environment.yaml
SHELL ["/bin/bash", "-c"]
# For some reason ibllib.io.video needs opencv which requires libgl1-mesa-dev ¯\_(ツ)_/¯
RUN apt update && apt install -y wget git libgl1-mesa-dev
RUN wget -O Mambaforge.sh  "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
RUN bash Mambaforge.sh -b -p /opt/conda && rm Mambaforge.sh
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
 mamba install conda-build &&\
 mamba env create -n iblenv --file=environment.yaml"
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh &&\
 conda activate iblenv &&\
 mamba install --yes pytorch pytorch-cuda=11.7 -c pytorch -c nvidia &&\
 conda clean --all -f -y"
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh &&\
 conda activate iblenv &&\
 pip install globus-sdk iblutil ibl-neuropixel ONE-api phylib pynrrd slidingRP &&\
 git clone https://github.com/int-brain-lab/iblapps.git &&\
 conda develop ./iblapps &&\
 git clone https://github.com/int-brain-lab/ibllib &&\
 conda develop ./ibllib &&\
 git clone https://github.com/berkgercek/neurencoding &&\
 conda develop ./neurencoding"
# The below allows interactively running the container with the correct environment, but be warned
# that this will not work with commands passed to the container in a non-interactive shell.
# In the case of e.g. `docker run thiscontainer python myscript.py`, the environment will not
# be activated and the script will fail. You will need to directly call the python executable
# in the container, e.g. `docker run thiscontainer /opt/conda/envs/iblenv/bin/python myscript.py`
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate iblenv" >> /root/.bashrc
RUN chmod -R 777 /root/
ENV BASH_ENV=/root/.bashrc
# Copying this repo in at the end so rebuilding the container with the latest code is easy. If you
# need to add your own dependencies, add them above this line so you won't have to reinstall them
# every time you rebuild the container. Alternatively just add them to the environment.yaml file.
