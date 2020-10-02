g++ FeedForwardNN.cpp Activation.cpp OutputMLP.cpp OscLayer.cpp DataLoader.cpp random_num_generator.cpp -fopenmp -I/usr/src/vcpkg/packages/matplotlib-cpp_x64-linux/include -I/usr/bin/python -lpython3.8 -L"." -lLibPyin -std=c++11 -o main.bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
