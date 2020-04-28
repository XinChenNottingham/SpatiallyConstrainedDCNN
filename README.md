# SpatiallyConstrainedDCNN

Please cite our papr if you find the code useful.

  N. Zhang, et al. A Spatially Constrained Deep Convolutional Neural Network for Nerve Fiber Segmentation in Corneal Confocal Microscopic Images Using Inaccurate Annotations, International Symposium on Biomedical Imaging, 2020. In press. 

A video presentation is availabe here: https://youtu.be/yOSiodu9mo8 
  
Contact xin.chen@nottingham.ac.uk

In this code, we only included a few example CCM images for testing the code."example_train.py" is the main function to train the model.

Semantic image segmentation is one of the most important tasks in medical image analysis. Most state-of-the-art deep learning methods require a large number of accurately annotated examples for model training. However, accurate annotation is difficult to obtain especially in medical applications. In this paper, we propose a spatially constrained deep convolutional neural network (DCNN) to achieve smooth and robust image segmentation using inaccurately annotated labels for training. In our proposed method, image segmentation is formulated as a graph optimization problem that is solved by a DCNN model learning process. The cost function to be optimized consists of a unary term that is calculated by cross entropy measurement and a pairwise term that is based on enforcing a local label consistency. The proposed method has been evaluated based on corneal confocal microscopic (CCM) images for nerve fiber segmentation, where accurate annotations are extremely difficult to be obtained. Based on both the quantitative result of a synthetic dataset and qualitative assessment of a real dataset, the proposed method has achieved superior performance in producing high quality segmentation results even with inaccurate labels for training.

## **Environment installation instruction:**
### **Windows 10**

1. Install [64-bit python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe) for windows or [other 64-bit versions](https://www.python.org/downloads/windows/) (select pip as an optional feature), 

1. Install NumPy, and Tensorflow 2.0 with GPU from PyPI:
    ```bash
    pip install --upgrade pip
    pip install --upgrade Numpy
    pip install --upgrade tensorflow-gpu
    ```

3. Install CUDA 10.0.

4. Download cuDNN and copy the files (bin, include, lib) from `*\cuda\` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\`.

5. Test Tensorflow 2.0:
    ```python
    import tensorflow as tf
    hello = tf.constant("hello TensorFlow!")
    print (hello)
    ```
---

### **Linux (Server)**

1. install and update Anaconda

    ```bash
    # move to local folder
    mkdir -p $HOME/usr/local/
    cd $HOME/usr/local

    # download and install Anaconda (if not installed)
    wget "https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh"
    bash Anaconda3-2019.10-Linux-x86_64.sh

    # remove install file
    rm Anaconda3-2019.10-Linux-x86_64.sh

    # update Anaconda
    conda update --all
    ```

2. Create conda environment for tensorflow
    ```
    conda create --name tf2_gpuenv
    ```

3. Activate the environment and install tensorflow-gpu
    ```bash
    # activate tensorflow environment

    conda activate tf2_gpuenv

    # install python 3.7 package (include pip)
    conda install python=3.7

    # install tensorflow-gpu use pip
    pip install --upgrade tensorflow-gpu

    # install cuda 10.0 and cudnn
    conda install cudatoolkit=10.0
    conda install cudnn
    ```

4. Choosing GPUs
    ```bash
    # check available gpus
    nvidia-smi

    # choose gpu [n]
    export CUDA_VISIBLE_DEVICES=[n]
    ```

5. Test tensorflow2 gpu version in python
    ```python
    import tensorflow as tf

    hello = tf.constant("hello TensorFlow!")
    print (hello)
    ```
---

### **Additional library needed:**
```bash
pip install --upgrade numpy, sklearn, scipy
pip install --upgrade opencv-python
pip install --upgrade Pillow
pip install --upgrade nibabel
pip install --upgrade shutil
```
