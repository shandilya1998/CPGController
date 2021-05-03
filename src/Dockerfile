FROM nvidia/cuda:11.0-base-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app && cd /app && git clone https://github.com/shandilya1998/CPGController
WORKDIR /app/CPGController/src

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.3 \
 && conda clean -ya

RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
 && sudo apt update \
 && sudo apt -y install ros-noetic-desktop-full \
 && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc \
 && echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)simulations/ws/src" >> ~/.bashrc \
 && source ~/.bashrc \
 && echo "source /opt/ros/noetic/setup.bash" >> ~/.bash_profile \
 && echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)simulations/ws/src" >> ~/.bash_profile \
 && source ~/.bash_profile \
 && sudo apt -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential \
 && sudo apt -y install python3-rosdep \
 && sudo rosdep init \
 && rosdep update \
 && sudo apt -y install ros-noetic-moveit ros-noetic-*-controllers \
 && cd src/simulations/ws \
 && catkin_make \
 && cd ../../../ \
 && sudo apt install -y python3-pip \
 && pip3 install tqdm numpy pandas matplotlib pybullet  tensorflow tf_agents

# CUDA 11.0-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=11.0.221 \
    "pytorch=1.7.0=py3.8_cuda11.0.221_cudnn8.0.3_0" \
    "torchvision=0.8.1=py38_cu110" \
 && conda clean -ya
