import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore all messages
import sys
import cv2
import re
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tabulate import tabulate


class Predict():
    def __init__(self, path="predict_data", checkpoint="checkpoints", labels=None, gui=0):
        self.path_data = path
        self.checkpoint_dir = checkpoint
        self.labels = labels
        self.gui = gui
        params = "python/predict_params.csv"
        self.df = pd.read_csv(params)

    # ============================================================
    def get_images(self, shape):
        def sorted_nicely(data):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(data, key=alphanum_key)

        if os.path.exists(self.path_data + "/insert your own images here that you want to predict.txt"):
            os.remove(self.path_data + "/insert your own images here that you want to predict.txt")
            time.sleep(0.5)
        dimx = shape[1]  # Get width of expected input shape
        dimy = shape[2]  # Get height of expected input shape
        images = []
        src_images = []
        data = os.listdir(self.path_data)
        if ".DS_Store" in data:  # Only necessary for MacOS
            os.remove(self.path_data + "/" + ".DS_Store")
            time.sleep(1)
            data = os.listdir(self.path_data)
        data = sorted_nicely(data)
        for file in data:
            try:
                image = cv2.imread(self.path_data + "/" + file)  # Read the image in open-cv
                src_img = plt.imread(self.path_data + "/" + file)  # Read the same image in Matplotlib
                image = cv2.resize(image, (dimx, dimy))  # Resize the open-cv image
                images.append(image)  # Append the resized image to a list
                src_images.append(src_img)  # Append the source image to a list
            except:
                print("=" * 100)
                print("ERROR! OpenCV could not open the file '{}'\n"
                      "This is probably due to an invalid character in the file name or the file is corrupted in some way.\nRename "
                      "the file and try again or check if there are folders or non image files in the directory: "
                      "\n'{}'.".format(file, self.path_data))
                print("=" * 100)
                sys.exit(1)
        if len(data) == 0:
            print("=" * 100)
            print("ERROR! Your selected folder is empty: '{}'".format(self.path_data))
            print("=" * 100)
            sys.exit(1)
        return images, src_images, data

    # ============================================================
    def preprocess_data(self, images, shape):
        def normalize(img, img_normalize, channels):
            if channels == 3:
                pass
            else:
                img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Grayscale image

            if img_normalize == "2":
                pass
            else:
                img = img / 255.0  # Normalize px values between 0 and 1
            return img

        img_normalize = str(self.df["img_normalize"][0])
        channels = shape[3]

        for x in range(len(images)):
            images[x] = normalize(images[x], img_normalize, channels)

        images = np.array(images)  # TF needs numpy array
        # Reshape images back to its original form of width * height. (Open-cv flips width and height when resizing an image)
        try:
            images = images.reshape(images.shape[0], images.shape[2], images.shape[1], channels)
        except:
            print("=" * 100)
            print("ERROR! Make sure you choose the correct folder. The predict folder must contain images only.\n"
                  "Subfolders within this directory are not allowed. Your selected path was: \n'{}'".format(self.path_data))
            print("=" * 100)
            sys.exit(1)
        return images

    # ============================================================
    def predict_data(self, src_images, predictions, data, file, preview="y", show_instance="y", counter=1):
        if preview == "n":  # No preview images if u predict all models
            inp = "n"
        else:
            inp = input("Show preview images y/n (default = 'y'): ")  # Ask the user if preview images should be shown
        if self.gui == 0:
            labels_file_name = self.df["csv_name"][0]
            df_labels = pd.read_csv("labels/" + labels_file_name)  # Read the labels file
        else:
            labels_file_path = self.labels
            df_labels = pd.read_csv(labels_file_path + "/labels_generated.csv")  # Read the labels file

        label_names = df_labels[self.df["csv_column"][0]].tolist()  # Extract the labels and append it to a list
        prediction_list = []
        probability_list = []
        label_names_str = []
        for label in label_names:
            label_names_str.append(str(label))  # Convert all labels to strings (Matplotlib displays only strings not numbers)

        if inp == "n":  # Don't show preview images
            for i in range(len(src_images)):  # Iterate over all images
                prediction_list.append(label_names_str[np.argmax(predictions[i])])  # Append the most likely prediction to a list
        else:  # Show preview images
            for i in range(len(src_images)):  # Iterate over all images
                image = src_images[i]
                plt.imshow(image)  # Show the source image
                plt.title("Prediction: " + label_names_str[np.argmax(predictions[i])])  # Insert the most likely prediction
                prediction_list.append(label_names_str[np.argmax(predictions[i])])  # Append the most likely prediction to a list
                plt.show()
        if show_instance == "n":
            self.df2["Prediction " + str(counter)] = prediction_list
        else:
            df = {"File": data, "Prediction": prediction_list}  # Create a dictionary showing every file and the corresponding prediction
            df = pd.DataFrame(df)
            print("\nPredictions for model file:", file)
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))  # Print the dictionary as a nice table

    # ============================================================
    def initialize(self):
        def sorted_nicely(l):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        if str(self.df["mode"][0]) == "3":
            # --- prevent TF from using more VRAM than the GPU actually has ---
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
        elif str(self.df["mode"][0]) == "2" or str(self.df["mode"][0]) == "4":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU Usage, instead of GPU

        data = os.listdir(self.checkpoint_dir)
        if ".DS_Store" in data:  # Only necessary for MacOS
            os.remove(self.checkpoint_dir + "/" + ".DS_Store")
            time.sleep(1)
            data = os.listdir(self.checkpoint_dir)
        if os.path.exists(self.checkpoint_dir + "/your model will be saved in this directory.txt"):
            os.remove(self.checkpoint_dir + "/your model will be saved in this directory.txt")
            time.sleep(0.5)
            data = os.listdir(self.checkpoint_dir)

        if len(data) > 1:  # Check if multiple models exist
            data = sorted_nicely(data)
            for i in range(len(data)):
                print("[{}]: '{}'".format(i + 1, data[i]))
            print("[{}]: Predict all models".format(len(data)+1))
            inp = input("There are multiple files in this directory. (Choose file with number between "
                        "'1' and '{}'; default = '{}'): ".format(len(data)+1, len(data)))

            if inp != str(len(data)+1):  # Predict a single model file
                if inp == "":  # If input = empty
                    file = data[len(data) - 1]  # Choose latest model
                else:
                    file = data[int(inp) - 1]  # Choose desired model
                model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)  # load the model using the load_model function from keras
            else:  # Predict all models
                self.df2 = {"File": None}  # Initialize a new dictionary
                counter = 1
                for file in data:  # Iterate over each model
                    model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)  # load the model using the load_model function from keras
                    config = model.get_config()  # Get the config from the model
                    shape = config["layers"][0]["config"]["batch_input_shape"]  # Get the expected input shape
                    images, src_images, data = self.get_images(shape)
                    images = self.preprocess_data(images, shape)
                    predictions = model.predict(images)
                    # probas = model.predict_proba(images)
                    self.predict_data(src_images, predictions, data, file, "n", "n", counter)
                    counter += 1
                self.df2["File"] = data
                df = pd.DataFrame(self.df2)
                print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))  # Print the dictionary as a nice table

                sys.exit(1)  # Exit program after all models are predicted
        else:  # There is only one model file in the directory
            file = data[0]
            model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)  # load the model using the load_model function from keras

        config = model.get_config()  # Get the config from the model
        shape = config["layers"][0]["config"]["batch_input_shape"]  # Get the expected input shape
        images, src_images, data = self.get_images(shape)
        images = self.preprocess_data(images, shape)
        predictions = model.predict(images)
        self.predict_data(src_images, predictions, data, file)
