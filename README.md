# LIDAR-enhanced-object-detection-for-vehicles

## Setup

### Install WSL
The Waymo opendataset only supports Linux. Here is an install guide:
https://docs.microsoft.com/en-us/windows/wsl/install
Use Ubuntu as the default distribution

### Clone the git repo into WSL and open it in vs code
**Make sure to do this as a user (instead of root)**

### Install WSL extension for vscode
https://code.visualstudio.com/docs/remote/wsl

### Install Cuda for WSL 2
https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#3-install-nvidia-cuda-on-ubuntu

### Install Anaconda in WSL
https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da

### Setup anaconda environment

```
# In the root folder of the git repo:
conda env create -n mti830 --file environment.yml
```

### Run the notebook
First, you need to install the jupyter notebook extension in vs code, then open training.ipynb. It should work now.
