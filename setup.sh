#!/bin/bash
sudo apt install -y ubuntu-drivers-common
sudo apt install linux-headers-$(uname -r)
sudo apt install -y nvidia-cuda-toolkit
nvcc -V
echo "Cuda Location>>>>>>>"
whereis cuda
tar -xvzf cudnn-10.1-linux-x64-v7.6.5.32.tgz
sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
sudo apt install -y nvidia-driver-460
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
pip3 install tqdm numpy pandas matplotlib pybullet  tensorflow tf_agents
cd src
sudo reboot
