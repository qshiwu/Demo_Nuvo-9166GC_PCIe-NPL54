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

- YOLOv8
```


sudo apt install -y libgtk2.0-dev pkg-config
sudo apt install -y qt6-wayland
sudo pip uninstall opencv-python opencv-python-headless -y
sudo pip uninstall opencv-python opencv-python
pip install opencv-python --no-cache-dir

pip install ultralytics opencv-python pillow torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



```