#!/bin/bash
update-alternatives --install /usr/bin/python python /usr/bin/python3 1
update-alternatives --install /usr/local/bin/python python /usr/bin/python 1
python -m pip install --upgrade --force pip
pip install -r requirements.txt
sudo apt-get update
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-catkin-pkg python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update
cd src/simulations/ws
sed -i 's#/home/shandilya/Desktop/CNS/DDP#/content/CPGController#g' build/CMakeCache.txt
sed -i 's#/home/shandilya/Desktop/CNS/DDP#/content/CPGController#g' build/Makefile
catkin_make
source devel/setup.bash
cd .. && cd ..
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/content/CPGController/src/simulations/ws/src
