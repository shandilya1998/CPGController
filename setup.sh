#!/bin/bash
sudo apt install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
echo "export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}" >> ~/.bash_profile
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bash_profile
sudo nvidia-smi
sudo apt install -y gdb
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt -y install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/shandilya/CPGController/src/simulations/ws/src" >> ~/.bashrc
source ~/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bash_profile
echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/shandilya/CPGController/src/simulations/ws/src" >> ~/.bash_profile
source ~/.bash_profile
sudo apt -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt -y install python3-rosdep
sudo rosdep init
rosdep update
sudo apt -y install ros-noetic-moveit ros-noetic-*-controllers
cd src/simulations/ws
catkin_make
cd ../../../
sudo apt install -y python3-pip
pip3 install -y tqdm numpy pandas matplotlib pybullet  tensorflow tf_agents
cd src
