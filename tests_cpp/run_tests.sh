rm -r build
mkdir build
cmake -S . -B build
make -C build
./build/ANNarchy_CPP_Templates
