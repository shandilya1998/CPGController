#!/bin/bash
echo "------------------------------"
echo "Starting Setup"
echo "------------------------------"
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo apt install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
echo "------------------------------"
echo "Installing Cuda"
echo "------------------------------"
sudo apt-get -y install cuda
echo "export PATH=/usr/local/cuda-11.2/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-11.2/bin:$PATH" >> ~/.bash_profile
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" >> ~/.bash_profile
echo "------------------------------"
echo "Installing cuDNN"
echo "------------------------------"
tar -xzvf cudnn-11.0-linux-x64-v8.0.2.39.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
echo "------------------------------"
echo "Installing gdb"
echo "------------------------------"
sudo apt install -y gdb
echo "------------------------------"
echo "Installing ROS"
echo "------------------------------"
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
echo "------------------------------"
echo "Installing Moveit & Controllers"
echo "------------------------------"
sudo apt -y install ros-noetic-moveit ros-noetic-*-controllers
cd src/simulations/ws
catkin_make
cd ../../../
echo "------------------------------"
echo "Installing Python Dependencies"
echo "------------------------------"
sudo apt install -y python3-pip
pip3 install tqdm numpy pandas matplotlib pybullet  tensorflow tf_agents
cd src
sudo reboot
