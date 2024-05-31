FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 torchtext==0.16.0 torchdata==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    tyro diffusers dearpygui einops accelerate lpips pygltflib rembg[gpu,cli] trimesh kiui xatlas roma plyfile \
    https://github.com/camenduru/wheels/releases/download/colab/curope-0.0.0-cp310-cp310-linux_x86_64.whl

RUN GIT_LFS_SKIP_SMUDGE=1 git clone -b dev --recursive https://github.com/camenduru/dust3r /content/dust3r

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -d /content/dust3r/checkpoints -o DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

COPY ./worker_runpod.py /content/dust3r/worker_runpod.py
WORKDIR /content/dust3r
CMD python worker_runpod.py
