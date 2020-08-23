g++ FeedForwardNN.cpp Activation.cpp OutputMLP.cpp OscLayer.cpp DataLoader.cpp random_num_generator.cpp -fopenmp -I/usr/src/vcpkg/packages/matplotlib-cpp_x64-linux/include -I/usr/local/include -I/usr/include/python3.8 -lpitch_detection -lffts -larmadillo -lmlpack -L/usr/src/pitch-detection  -lpython3.8 -std=c++11 -o main.bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
