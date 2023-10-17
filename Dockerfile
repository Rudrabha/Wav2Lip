FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# install python via pyenv
RUN apt-get update && apt-get install -y --no-install-recommends \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libffi-dev \
	liblzma-dev \
	git \
	ca-certificates \
    libgl1 \
	&& rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
ARG PYTHON_VERSION=3.8
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	pyenv install $PYTHON_VERSION && \
	pyenv global $PYTHON_VERSION

# install cog
RUN pip install cog

# install deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libsndfile1 \
	&& rm -rf /var/lib/apt/lists/*

# copy to /src
ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/elliottzheng/batch-face.git@master

# copy sources
COPY . .

ENV PYTHONUNBUFFERED=1

# run cog
CMD python3 -m cog.server.http
