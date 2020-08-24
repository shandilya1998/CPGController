FROM ubuntu:latest

# These commands copy your files into the specified directory in the image
# and set that as the working location
COPY . /usr/src/pitch-detection

# Install dependencies from apt
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
git \
cmake \
curl \
zip \
unzip \
tar \
vim \
python3-pip \
 python3-dev \
gcc \
g++ \
gdb \
libblas-dev \
liblapack-dev \
libboost-dev \
libarmadillo-dev \
libmlpack-dev \
libgtest-dev \
libbenchmark-dev \
&& cd /usr/local/bin \
&& ln -s /usr/bin/python3 python \
&& pip3 install --upgrade pip \
&& apt-get update \
&& pip install matplotlib


# Get and install ffts library
RUN cd /usr/src \
&& git clone https://github.com/anthonix/ffts.git ffts-repo \
&& cd ffts-repo \
&& mkdir build \
&& cd build \
&& cmake .. \
&& make install

RUN cd /usr/src \
&& git clone https://github.com/Microsoft/vcpkg.git \
&& cd vcpkg \
&& ./bootstrap-vcpkg.sh \
&& ./vcpkg integrate install \
&& echo "export PATH=/usr/src/vcpkg:${PATH}" >> /root/.bashrc \
&& ./vcpkg install matplotlib-cpp \
&& echo "export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}" >> /root/.bashrc 

# Build and install the pitch-detection library, as well as the tests and benchmarks
RUN cd /usr/src/pitch-detection && make clean all && make -C test clean all && make install

LABEL Name=pitch-detection Version=0.0.1
