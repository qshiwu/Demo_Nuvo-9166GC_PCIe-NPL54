# Demo_Nuvo-9166GC_PCIe-NPL54

## Prerequisites

- Refer to this [link](https://neousys.gitbook.io/nru-series/misc/one-page/nuvo-9166gc) to manually change to a good fan speed of Nuvo-9166GC

- Refer to this [link](https://neousys.gitbook.io/nru-series/misc/one-page/nuvo-9160gc/running-carla-on-nuvo-9160gc-with-ubuntu-22.04) to install nvidia GPU driver

- Refer to this [link](https://neousys.gitbook.io/nru-series/misc/one-page/pcie-npl54/getting-started) to install PCIe-NPL54

- Install CUDA by
```
sudo apt-get install -y nvidia-cuda-toolkit
```

- Install Notebook (to read ipynb)
```
pip install notebook
```

- SAM2 installation
The SAM2 checkpoints files are too large, which will not be backuped in this Repo

```
# Install SAM2
git clone https://github.com/facebookresearch/sam2.git && cd sam2
sudo apt-get install -y python3-pip
sudo pip install -e .
sudo pip install -e ".[notebooks]"

## Download Checkpoints
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

- Pyhton OpenCV with GUI
```
sudo pip uninstall opencv-python opencv-python-headless -y
sudo pip uninstall opencv-python opencv-python
pip install opencv-python --no-cache-dir
```

- Movenet Installation
# Ref: https://github.com/lee-man/movenet-pytorch.git
```
pip install tensorflow
pip install tensorflow-hub
pip install tensorflow-docs
pip install torch torchvision

```

- YOLOv8 Installation
```
sudo apt install -y libgtk2.0-dev pkg-config
sudo apt install -y qt6-wayland

pip install ultralytics opencv-python pillow torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Posenet on Pytorch
```
cd ~/Desktop
git clone https://github.com/michellelychan/posenet-pytorch.git
cd posenet-pytorch/
python3 get_test_images.py
python3 image_demo.py --model 101 --image_dir ./images --output_dir ./output
```

A quick example
```
python3 posenet_image_demo.py --model 101 --image_dir ./images --output_dir ./output
```


- MobileSAMv2
https://github.com/ChaoningZhang/MobileSAM
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM
pip install gradio


## Build OpenCV with CUDA and GUI support

# Install CUDA and CUDNN

Ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install cuda-toolkit
sudo apt-get -y install cudnn9-cuda-12
```

```

## Make sure the headless openCV is uninstalled
pip show opencv-python
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python


cd ~/Desktop
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git


cd opencv
mkdir build
cd build

export CC=/usr/bin/gcc-10; export CXX=/usr/bin/g++-10

cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D OPENCV_DNN_CUDA=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_OPENCV_WORLD=OFF \
-D CUDA_ARCH_BIN=8.0 \
-D OPENCV_EXTRA_MODULES_PATH=/home/dvt/Desktop/opencv_contrib/modules/ /home/dvt/Desktop/opencv/ \
-D BUILD_EXAMPLES=OFF \
-D HAVE_opencv_python3=ON \
..


make -j8
sudo make install

```

