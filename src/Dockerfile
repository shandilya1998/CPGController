FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
WORKDIR /app
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata keyboard-configuration
RUN apt-get update && \
    apt-get install -y lsb-release \
        curl \
        ca-certificates \
        sudo \
        gdb \
        git \
        vim \
        bzip2 \
        libx11-6 && \
    apt-get clean all

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full
RUN apt -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-rosdep && \
    rosdep init && rosdep update && \
    apt -y install ros-noetic-moveit ros-noetic-*-controllers && \
    apt install -y python3-pip && \
    pip3 install tqdm numpy pandas matplotlib pybullet tensorflow tf_agents && \
    pip3 install torch==1.8.1+cu111 torchsummary torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# setup entrypoint
COPY ./ros_entrypoint.sh /
RUN bash -c "chmod +x /ros_entrypoint.sh"

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

