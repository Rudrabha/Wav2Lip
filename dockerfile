FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN mkdir /workspace/Wav2Lip
WORKDIR /workspace/Wav2Lip

# uncomment this if you have trouble downloading packages
# RUN rm /etc/apt/sources.list.d/cuda.list && \
#     rm /etc/apt/sources.list.d/nvidia-ml.list && \
#     sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
#     sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
#     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get -y update --fix-missing && \
    apt-get install -y libglib2.0-0 libxrender1 ffmpeg libsm6 && \
    pip install librosa==0.7.0 opencv-contrib-python==4.6.0.66

