# Automated-CNN-Image-Classifier-TF
Fully automated TF2 CNN image classifier to automatically train and predict images given a dataset of your choice



- [Introduction](#introduction)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Train a model](#train-a-model)
    - [Augmentation options](#augmentation-options)
    - [Label options](#label-options)
    - [Preprocessing options](#preprocessing-options)
    - [Training options](#training-options)
    - [Model options](#model-options)
    - [Executions options](#execution-options)
- [Predict data](#predict-data)
- [Change the predefined model structure](#change-the-predefined-model-structure)

## Introduction
The goal of this project is to provide an easy way to train and predict a CNN in TF. 
It is meant to be used by beginners who just started their data science journey and for people who quickly want to train a decent performing image classifier with minimal effort. 
Some key features are:
- GUI
- Augmentation of images of your choice via a few inputs
- Choose different preprocessing options
- Build the model via a few inputs
- Choose preferred execution mode

## Prerequisites
Make sure you have [Anaconda](https://docs.anaconda.com/anaconda/install/) installed.
- Windows:
    - Install the windows version of Anaconda [here](https://docs.anaconda.com/anaconda/install/windows/)
    - Make sure you add conda to PATH (or you can't use conda commands from CMD)
- Mac: Install the MacOS version of Anaconda [here](https://docs.anaconda.com/anaconda/install/mac-os/)
- Linux: Install the Linux version of Anaconda [here](https://docs.anaconda.com/anaconda/install/linux/)
    

## Installation
1. Open command line and navigate to a directory of your choice. 
2. Clone this repository to your machine via the following command:
``` shell
git clone https://github.com/Daniel-Fauland/CNN_Image_Classification_Generator.git
```
3. Navigate to the cloned repository:
``` shell
cd CNN_Image_Classification_Generator
```
**Windows:**

4. Install [**requirements_windows.yml**](requirements_windows.yml). This will created a new virtual environment that is separated from you existing python installation and therefore will not interfere with any of your other packages:
    ``` shell
    conda env create -f requirements_windows.yml
    ```
5. Activate the new virtual environment:
    ``` shell
    conda activate tf
    ```
6. _Optional:_
    - If you have a NVIDIA GPU and you want to use the GPU for training and/or predicting you have to download additional drivers
    - You need a free NVIDIA developer [account](https://developer.nvidia.com/developer-program)  
    - You need to download and install [NVIDIA CUDA 11.1.0](https://developer.nvidia.com/cuda-toolkit-archive)
    - Download [cuDNN v8.0.5](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 11.1
    - Watch installation guide [here](https://youtu.be/hHWkvEcDBO0?t=235)
    - Make sure you download the specific version given above and not an older/newer version because there is a high chance that it will not work

   
**MacOS:**
4. Create a new virtual environment:
    ``` shell
    conda create --name tf python=3.8.5
    ```
5. Activate the new environment:
    ``` shell
    conda activate tf
    ```
6. Install [**requirements_mac.txt**](requirements_mac.txt). This will install all necessary packages for this project:
    ``` shell
    pip install -r requirements_mac.txt
    ```

<br />
After the installation is complete you can run it from your command line:

- You can use the GUI by typing:
``` shell
python start_gui.py
```
- Or you can use the command line version:
``` shell
python train_model.py
python predict_data.py
```

You can also open this project in an IDE (e.g. PyCharm): 
- Open PyCharm > Go to File > Open > Select CNN_Image_Classification_Generator
- Go Settings > Project: CNN_Image_Classification_Generator > Python Interpreter
- Click the dropdown menu and click 'Show all'
- Select the Conda env 'tf' and click 'Apply'
- (If it's not listed click the '+' icon > Conda Environment > Existing environment > Interpreter > Choose the environment folder)


## Train a model
Make sure your training data and labels file (if you have one) fulfill the following conditions:
- You have one folder containing all the images for each category (no sub folders within a category)
- You do not have a separate folder with validation images
- The folder and file names contain only the following characters: **'a-z A-Z 0-9 _ - + . \*'**
- The categories in your labels file appear in the same order as the alphabetically orderd folders in 'training_data' --> The category name for the third folder is in the third line in your labels file and the category name for the fith folder is in the fith line in your labels file, etc.
   
Run the [**start_gui.py**](start_gui.py) (GUI verison) or [**train_model.py**](train_model.py) (command line version) file. <br />
You can select a background image for the gui by changing the value of the variable img in [**start_gui.py**](start_gui.py) <br />
**Note:** By default all info, logs and warnings are deactivated. They can be turned back on if you change the following statement in line 2 in file 
[**augmentation.py**](python/augmentation.py), [**preprocess.py**](python/preprocess.py), [**model.py**](python/model.py) and [**predict.py**](python/predict.py):
``` python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Change this value to 0
```

### Augmentation options
[Image augmentation](https://towardsdatascience.com/image-augmentation-14a0aafd0498) is a technique that increases the amount of training images by changing some aspects of the image and saving it as a separate file.
You have various options for image augmentation:
- \[2]: 90 degrees rotation = [Rotate](https://www.tensorflow.org/api_docs/python/tf/image/rot90) an image by 90 degrees to the right
- \[3]: 180 degrees rotation = [Rotate](https://www.tensorflow.org/api_docs/python/tf/image/rot90) an image by 180 degrees
- \[4]: 270 degrees rotation = [Rotate](https://www.tensorflow.org/api_docs/python/tf/image/rot90) an image by 270 degrees to the right
- \[5]: Randomly flip image to the left or right --> Image will randomly be [flipped to the left or right](https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right)
- \[6]: Randomly flip up or down --> Image will randomly be [flipped up or down](https://www.tensorflow.org/api_docs/python/tf/image/random_flip_up_down)
- \[7]: Randomly change hue --> The [hue](https://www.tensorflow.org/api_docs/python/tf/image/random_hue) of the image will randomly be changed based on a delta of **0.1**
- \[8]: Randomly change saturation --> The [saturation](https://www.tensorflow.org/api_docs/python/tf/image/random_saturation) of the image will randomly be changed between **0.6** and **1.6**
- \[9]: Randomly change brightness --> The [brightness](https://www.tensorflow.org/api_docs/python/tf/image/random_brightness) of an image will randomly be changed based on a delta of **0.05**
- \[10]: Randomly change contrast --> The [contrast](https://www.tensorflow.org/api_docs/python/tf/image/random_contrast) will randomly be changed between **0.7** and **1.3**

You can choose one or more options by typing the displayed number separated by space. For example the input '2 5 7' will rotate the images by 90 degrees, randomly flip left or right and randomly change the hue.
Type '1' or click enter to skip the augmentation. You can also delete your augmentations by typing '11'. <br />
The values of hue, saturation, brightness and contrast can be changed in [**augmentation.py**](python/augmentation.py).

**GUI version:** Select the augmentations via checkboxes.

### Label options
A label file is used for assigning your folders names in form of a string. You can either provide a labels file in form of a csv or create a labels file with a built-in function in the code.
If you choose to create the labels file via the built-in function you have to provide a name for each folder. Alternatively you can press enter without providing a name. 
In this case the label name for the folder will be the same as the folder name itself.

**GUI version:** A table will be shown where you can enter the label names.

### Preprocessing options
The preprocessing transforms the data in a way that TF accepts the data as a valid input. 
1. Resize all images to a specific shape: All images need to be the same shape before they can be passed to the model.
Type e.g. '64 32' to  resize all images to 64px width and 32px height. Alternatively you can just press to enter to the resize all images to the shape
   of the first image found.
   
2. [Grayscale](https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html) images: You can choose if you want to transform all images to black and white by typing '1' or pressing enter. 
   By typing '2' all images will be unchanged.
   
3. [Normalization](https://en.wikipedia.org/wiki/Normalization_(image_processing)) of images: Normalize the pixel values of all images by typing '1' or pressing enter. 
   The pixel value of normalized images only range from **0** to **1** instead of **0** to **255**.
   
   
**GUI version:** Select your Preprocessing options via checkboxes.
   
### Training options
In training options you can adjust 3 major settings:
1. Validation size: Validation size specifies how much % of a data is used to test the model accuracy and loss.
Type e.g. '20' for a validation size of 20 %.
   
2. Number of epochs: Choose how often you want to iterate over all training images. 
   If this number is to small your model will [underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) but if the number is too high on the other hand your model will likely [overfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit).
   
3. Batch size: The batch size specifies how many images you pass to the model at once before the model weights are updated. A higher batch size can increase the training speed as well as the model quality at the cost of more ram.
Keep in mind that a batch size that is too high can lead to worse generalization in some cases. This means that the highest possible batch size is not necessarily the best option.
   You can read more about batch size in [this article](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/).
   
**GUI version:** You can enter a number for these options via an input box.

### Model options
You can either define the model structure yourself (type '1' or press enter) or use the predefined model structure (type '2'). The model structure of the chosen model will be saved in the file [**model_summary.txt**](python/model_summary.txt) when the training starts.
The structure of the predefined model can be viewed in the file [**predefined_model_summary.txt**](python/predefined_model_summary.txt) . It has the following structure:
- Convolutional layer (num_neurons = 32, strides = (3,3), activation = "relu"))
- MaxPooling layer (pool_size = (2, 2))
- Convolutional layer (num_neurons = 64, strides = (3,3), activation = "relu"))
- MaxPooling layer (pool_size = (2, 2))
- Flatten layer
- Dropout layer (dropout_rate = 0.2)
- Hidden layer (num_neurons = 64)
- Dropout layer (dropout_rate = 0.2)

When defining your own model structure you can add up to four convolutional and max pooling layers, a dropout layer before the first hidden layer, 
any desired amount of hidden layers, and a dropout layer before the output layer.
- You can specify the number of neurons, the strides and the activation function for the [convolutional layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).
- You can specify the pool size for the [MaxPooling layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).
- You can specify the dropout rate (in %) for the [dropout layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout).
- You can specify the number of neurons as well as the activation function for the [hidden layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

### Execution options
Depending on your installation you might want to choose the execution mode yourself. That's why you have four different options available. Type '1' or press enter to choose mode 1:
- \[1]: Automatic = Do not specify any execution mode. TF will choose a mode based on your installation automatically.
  (If you have an NVIDIA GPU and CUDA installed, and you get a TF error by running on automatic mode try one of the other options)
- \[2]: GPU for training and CPU for predicting = Enables memory growth for GPU which should eliminate most errors with the GPU version of TF. CPU is used for predicting images
- \[3]: GPU for training and predicting = Same as option \[2] but also uses GPU for predicting. (**Note:** Predicting needs very little performance and is pretty fast most of the time, but the initialization of CUDA takes a few seconds.
  That is why option \[2] is the better execution mode in most situations.)
- \[4]: Force CPU for training and predicting = If you have any problems whatsoever with your GPU, use this mode.

**GUI version:** You can select your preferred option via a dropdown menu.

## Predict data
**Command line version:** Put the images that you want to predict inside the *predict_data* folder and run [**predict_data.py**](predict_data.py).

**GUI version:** Select the path to the images you want to predict as well as the checkpoint path and labels path.

**Make sure your prediction data fulfills the following conditions:**
- You put only images insides the *predict_data* directory / your desired directory if you use the GUI version.
- All images are either **'.png'** or **'.jpg'** or **'.jpeg'**
- The file names contain only the following characters: **'a-z A-Z 0-9 _ - + . \*'**
- There are no subfolders within this directory.


## Change the predefined model structure
The predefined model structure can be changed in [**predefined_model.py**](python/predefined_model.py)