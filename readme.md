# **Towards Secure and Usable 3D Assets: A Novel Framework for Automatic Visible Watermarking (WACV 2025 Oral)**  

📄 **Paper:** [arXiv 2409.00314](https://arxiv.org/pdf/2409.00314)  
🖥️ **Conference:** WACV 2025 (Oral)  
📂 **Code Repository:** [GitHub Link](#)  
👨‍💻 **Authors:** Gursimran Singh, Tianxi Hu, Mohammad Akbari, Qiang Tang, Yong Zhang  
🏢 **Huawei's AI Gallery Notebook:** [Link](https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=58b799a0-5cfc-4c2e-8b9b-440bb2315264)  
📹 **Video Samples:** [Download](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/3dwatermark/video_demo.mp4)  


## Framework
<p align="center">
<center>
<img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/3dwatermark/framework.png" alt="alt text" width="1000">
</center>
</p> 

## Download and extract the code


```python
!wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/3dwatermark/code.zip 
!unzip -qo code.zip
```

## Setting up environment
- To start with, confirm the base environment uses `Python 3.10.12` and `Ubuntu 22.04.2 LTS`. Note that older Linux distribution might not support all packages required, and gpu-supported versions for packages may differ depending on your gpu setup. 

- Create conda environment with:  


```python
!conda create -n "3dwatermark" python=3.10.12 ipython
!conda activate 3dwatermark
```

- Before installing anything else, install the blender python library:  


```python
!pip install bpy==4.0.0
```


```python
- Install `pytorch3d==0.7.6`:  
!pip install --no-index --no-cache-dir pytorch3d --trusted-host dl.fbaipublicfiles.com -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt221/download.html
```

- Install other required packages:  


```python
!conda env update --file conda_env.yml
!conda install --name 3dwatermark --file spec-file.txt
```

## Watermark 3D Models
- GPU is required for efficient implementation and the default device is `cuda:0`. The following command can be used to verify the number of GPUs available:


```python
!python -c "import torch; num_of_gpus = torch.cuda.device_count(); print(num_of_gpus)"
```

- To run the watermarking pipeline with the default example models and parameters, simply run the following:


```python
!python main.py
```

- Run with watermark embossing:  


```python
!python main_curvefollow.py
```
