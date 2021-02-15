#!/bin/bash
echo Starting ROS install
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

sudo apt update

sudo apt install ros-melodic-desktop-full
sudo apt install -y ros-melodic-rosmaster

pip install --extra-index-url https://rospypi.github.io/simple/ rospy
sudo apt install python-rosdep python-catkin-pkg python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update

echo ROS install done

printf -- '#!/bin/bash\n' > ros_start.sh
printf -- 'source /opt/ros/melodic/setup.bash\n' >> ros_start.sh
printf -- '/opt/ros/melodic/bin/roscore' >> ros_start.sh
chmod +x ros_start.sh
#cat ros_start.sh

echo Setting up Quadruped package
cd CPGController/src/simulations/ws
sed -i 's#/home/shandilya/Desktop/CNS/DDP#/content/CPGController#g' build/CMakeCache.txt
sed -i 's#/home/shandilya/Desktop/CNS/DDP#/content/CPGController#g' build/Makefile
catkin_make
source devel/setup.bash
cd ../../../
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/content/CPGController/src/simulations/ws/src
echo Package setup done
