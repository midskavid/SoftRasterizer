sudo apt-get update
sudo apt-get upgrade
sudo apt install --assume-yes build-essential cmake git pkg-config unzip ffmpeg qtbase5-dev python-dev python3-dev python-numpy python3-numpy
sudo apt install libhdf5-dev
sudo apt install --assume-yes libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev
sudo apt install --assume-yes libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt install --assume-yes libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev
sudo apt install --assume-yes libvorbis-dev libxvidcore-dev v4l-utils

cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.3.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.3.zip
unzip opencv.zip
unzip opencv_contrib.zip

cd ~/opencv-3.4.3/
mkdir build
cd build

# cmake ../ -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.3/modules -D BUILD_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_EXAMPLES=ON -D BUILD_PROTOBUF=OFF -D PROTOBUF_UPDATE_FILES=ON -D BUILD_opencv_dnn=OFF ..

cmake ./ -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D CUDA_GENERATION=Auto \
    -D WITH_CUBLAS=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_TIFF=ON \
    -D ENABLE_CXX11=ON \
    -D WITH_PROTOBUF=OFF \
    -D BUILD_opencv_legacy=OFF \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.3/modules \
    -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..


make -j8
if [ ! $? -eq 0 ]; then
    make clean
    make
fi

sudo make install
