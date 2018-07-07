# CK-Plus

Keras implementation of C3D model on Extended Cohn-Kanade Dataset (CK+). 

C3D is a neural network that introduces 3D convolutions and pooling layers. For more information please refer to the original paper:
Tran, Du, et al. "[Learning Spatiotemporal Features With 3D Convolutional Networks](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)." Proceedings of the IEEE International Conference on Computer Vision. 2015.


The user may define the following parameters upon calling the python script:

- Whether to use face extraction (using MTCNN) on the original images as a preprocessing step or not.
- Temporal depth of the image sequence transformed into a single instance for the 3D convolutions.
- Minimum sequence length of the video Sessions that are considered for training. By a combination of depth and this variable, the user defines the percentage of frames labeled per Session. For example, for a depth = 3 and minimum sequence length = 9, only Sessions with 9 frames or more are considered. In this example, the first 3 frames per sequence are labeled as neutral and the last 3 as the corresponding emotion. 
- Size into which the images will be resized to be feeded to the model.
  
---

# Requirements

Please make sure to comply with everything mentioned in this section before execution.

## Dataset

The code requires the user to specify the location of four folders: 

- **main_dir:** folder containing the three subfolders below.
- **labels_main_dir:** original ck+ folder with emotion labels.
- **images_main_dir:** original ck+ folder with images.
- **emotions_main_dir:** new folder in which images will be separated by emotion.

A suggested folder structure (following the above order) is:

- ~/ckp/
- ~/ckp/emotion_labels/
- ~/ckp/cohn-kanade-images/
- ~/ckp/emotion_images/

To get the database, please refer to the download site: http://www.consortium.ri.cmu.edu/ckagree/.

## Packages

Make sure to install the necessary packages:

- matplotlib
- numpy
- mtcnn
- opencv (>3.4)
- scikit-learn
- keras

# Running the code

To train the network, the following steps must be followed.

Run create_dataset.py defining the following arguments:

      python create_dataset.py \
      main_dir \ 
      labels_main_dir \ 
      images_main_dir \
      emotions_dir crop \
      crop_faces \
      neutral_label \
      min_seq_len \
      depth \ 
      t_height \
      t_width

For example:

    python create_dataset.py ~/ckp/ ~/ckp/emotion_labels/ ~/ckp/cohn-kanade-images/ ~/ckp/emotion_images/ 1 0 9 3 112 112

This script will add a folder per emotion to ~/ckp/emotion_images/ and fill them with Numpy binary files. Each file will have an image sequence with the shape (depth, t_height, t_width, channels) = (3, 112, 112, 3). For every file added to an emotion folder, another one will be added to the neutral folder. 
