# Real-time Face Recognition with a Light-weight Model

This repository implements a real-time face recognition system using a lightweight model based on GhostFaceNetV2 architecture. The model achieves state-of-the-art performance and is designed to run efficiently on CPU, making it suitable for local deployment without the need for a GPU.

Links to all mentioned datasets and model weights can be found at the bottom of the page.

# Image Preprocessing

For datasets, I have developed a module for alignment using RetinaFace for face and landmarks detection. RetinaFace outputs bounding boxes and 5 facial landmarks. In my alignment pipeline, I used `cv2.warpAffine` to rotate faces based on these landmarks. All datasets have been preprocessed using this script, ensuring consistent cropping and alignment.

![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/c4dfdbd7-6387-4a9d-aed3-8826758a2b81)


# GhostFaceNetV2

[GhostFaceNetV2](https://ieeexplore.ieee.org/document/10098610) is a lightweight convolutional neural network (CNN) with a backbone derived from MobileNet. Its main feature is the use of attention mechanisms. The architecture is based on depth-wise and point-wise convolutional blocks, ensuring efficient and effective performance.

This model closely approaches state-of-the-art performance while containing approximately 4 million parameters. This enables running the model in face recognition pipelines without the need for a GPU.

![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/9f5e1984-1590-4213-be64-ad3473a6fff9)


Original TensorFlow [implementation](https://github.com/HamadYA/GhostFaceNets)

## Training

### Train notebook
For experimentation and similar purposes, we provide a [notebook](https://github.com/wh1t3tea/face-recognition/blob/main/examples/arcface-training.ipynb) with a complete pipeline for training and testing the model.

![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/3064adcb-47ef-44d0-ab62-7739b9d796eb)
### Train Configuration

We provided a detailed train configuration file (`config.json`) with adjustable parameters such as learning rates, batch size, and optimizer settings.
For our training setup, we utilized the Stochastic Gradient Descent (SGD) optimizer with an initial learning rate of 0.1. The training process spanned 30 epochs, with a batch size of 256. Additionally, we employed a MultiStepLR scheduler with a step size of 3 and a gamma value of 0.1 to adjust the learning rate at specific intervals during training.
In the ArcMargin loss configuration, we incorporated a margin value of 0.5 and a scale factor of 32. These parameters play a crucial role in shaping the loss function's behavior, specifically in enhancing the discrimination between classes during training. The margin value introduces angular margins between different classes, while the scale factor adjusts the magnitude of feature embeddings, ultimately contributing to improved face recognition performance.

The complete training configuration used to train the model is available at the following [link](https://github.com/wh1t3tea/face-recognition/blob/main/recognition/cfg/config.json)


## Dataset

The model was trained on the open-source face-dataset:
- [Casia-WebFace dataset](https://arxiv.org/abs/1411.7923) - 0.5 million images/10k persons.
This dataset comprises images of 10,572 individuals. While it is a refined iteration of the Casia dataset, it still exhibits class imbalance. To address this, we advise implementing the [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) technique, which helps enhance class stability by appropriately weighting samples during the training process.

## ArcFaceLoss

The model was trained using ArcFaceLoss, a sophisticated loss function tailored for face recognition tasks. ArcFaceLoss enhances the discriminative power between classes by adding an angular margin to the cosine similarity between feature embeddings and class centers. This margin encourages the model to learn more separable feature representations, resulting in improved performance in face recognition tasks.

## Metrics

The repository provides the trained model weights, with the following metrics:
- 98.9% accuracy on the [LFW benchmark](https://paperswithcode.com/sota/face-recognition-on-lfw).
- TAR@FAR=0.01 - 0.91 on [Celeba500](https://www.kaggle.com/datasets/wannad1e/celeba-500-label-folders).

## Recognition
The key feature of this model's architecture is its lightweight nature, making it ideal for local deployment of face recognition systems on a CPU. Without using CUDA, the pipeline with the RetinaFace face detection model (S size) achieves 12-15 fps on a consumer-grade 6-core processor.
GPU inference allows to reach more then 30 fps.

## How to Use

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
2. **Create your config.json**:

   ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/65f4153b-73d3-44e9-9538-800180f73a28)
   

4. **Train the Model**:

    To train the model, run the `train.py` script, specifying the path to your config file:

    ```bash
    python train.py --config cfg/config.json
    ```
## Desktop Application

In addition to the face recognition model, we offer a user-friendly desktop application for real-time face recognition. This application provides a convenient interface for running face recognition locally on your machine.

![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/e9aee5fb-73d4-480c-a4c9-95e753e02a0c)

## Features

- **Real-time Face Recognition**: Experience seamless face recognition with live camera feed integration.
- **User-Friendly Interface**: Intuitive and easy-to-use interface for effortless navigation and operation.
- **Customizable Settings**: Adjust your face dataset.
- **Cross-Platform Compatibility**: Compatible with Windows, macOS, and Linux operating systems for versatile usage.

## Installation

To use the desktop application, follow these simple steps:

1. **Clone the Repository**: Clone the repository to your local machine by running the following command:

    ```bash
    git clone https://github.com/wh1t3tea/face-recognition
    ```

2. **Navigate to the Desktop App Directory**: Move to the directory containing the desktop application files:

    ```bash
    cd /desktop
    ```

4. **Run the Application**: Launch the desktop application by running:

    ```bash
    python desktop_app.py
    ```

## Usage

Once the application is running, you can perform the following actions:

- **Adjust your own face dataset**: Click on:


  ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/de691f96-683f-46c8-b555-17748e00d4d5)

- **Face photo database**: The face photo database should be organized in the following format: each individual should have their own folder named after them, containing photos of that individual:
  
   ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/806cc1ec-56de-48ae-abd7-19739506adac)
-> ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/d00e0d31-b831-4aa3-8362-25f3cac21379)
-> ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/85787309-a672-4295-aed5-56fbd0417d1f)

- **Start Face Recognition**: Click on the "Start" button to initiate real-time face recognition:
  
  ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/70f4947f-5809-40ef-a3e0-53688c28175f)

- **Adjust Settings**: Modify settings such as face detection threshold, recognition confidence level, device and camera resolution to optimize performance:

  ![image](https://github.com/wh1t3tea/face-recognition/assets/128380279/1521d3fc-0175-4ab9-911e-1a18f897d670)

   If you intend to use GPU acceleration within our application, make sure to install the necessary CUDA drivers and cuDNN libraries. These components are vital for unlocking GPU capabilities and maximizing performance.
  
- **View Recognition Results**: View recognized faces and corresponding labels in the application interface in real-time. You can also adjust your callbacks:

  <div>
    <img src="https://github.com/wh1t3tea/face-recognition/assets/128380279/6da070d7-c5fe-4a49-b5f4-deab1730f7b1" alt="Image 1" style="width: 45%;">
    <img src="https://github.com/wh1t3tea/face-recognition/assets/128380279/5c39460a-9752-46f7-a80c-a439107c352f" alt="Image 2" style="width: 45%;">
  </div>

## Model weights
- [GhostFaceNetV2](https://github.com/wh1t3tea/face-recognition/blob/main/ghostfacenet_v2_4_1.pth)

## Train logs
- [train-logs](https://www.dropbox.com/scl/fi/smbfu9q9f6zn0424yg2i8/train.log?rlkey=xnwp2r6r27yu94fxkvapgc6hq&st=4rzjs3if&dl=0)

## Datasets
- [Casia-WebFace_aligned](https://www.kaggle.com/datasets/wannad1e/casia-aligned)
- [VGGface2](https://www.kaggle.com/datasets/wannad1e/vggface2)
- [Celeba500](https://www.kaggle.com/datasets/wannad1e/celeba-500-label-folders)
- [LFW-id-rate](https://www.kaggle.com/datasets/wannad1e/ssssas)
- [LFW-pair-benchmark](https://www.kaggle.com/datasets/wannad1e/lfw-benchmark)
## References

- [GhostNet](https://arxiv.org/abs/1911.11907)
- [GhostFaceNets](https://ieeexplore.ieee.org/document/10098610)
- [ArcFaceLoss](https://arxiv.org/abs/1801.07698)
- [Casia-WebFace dataset](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
- [GhostFaceNet-TF](https://github.com/HamadYA/GhostFaceNets)
- [PyTorch Implementations](https://github.com/Hazqeel09/ellzaf_ml)

## Notes

- This project is developed for educational purposes and can be further refined and expanded for specific needs.
- When using this code in your projects, please provide references to the original research papers and datasets.




