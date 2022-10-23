# VFR-UFD: Reconstructing Unsteady Flow Data From Representative Streamlines via Diffusion and Deep-Learning-Based Denoising
This repository contains the PyTorch implementation for paper "Reconstructing Unsteady Flow Data From Representative Streamlines via Diffusion and Deep-Learning-Based Denoising". 

# Prerequisites
* Linux
* CUDA >= 10.0
* Python >= 3.7
* Numpy
* Pytorch >= 1.0

# How to run the code
* First, change the directory path and parameter settings (e.g., batch size, dimensions of your data etc.). 
* Second, to train the model, comment line 679 to line 702 and simply call python3 main.py. 
* Third, to inference the new data, comment line 661 to line 672 and uncomment line 679 to line 702 and simply call python3 main.py.

# Citation
@article{gu2021reconstructing,<br/>
  title={Reconstructing Unsteady Flow Data From Representative Streamlines via Diffusion and Deep-Learning-Based Denoising},<br/>
  author={Gu, Pengfei and Han, Jun and Chen, Danny Z and Wang, Chaoli},<br/>
  journal={IEEE Computer Graphics and Applications},<br/>
  volume={41},<br/>
  number={6},<br/>
  pages={111--121},<br/>
  year={2021},<br/>
  publisher={IEEE}<br/>
}
