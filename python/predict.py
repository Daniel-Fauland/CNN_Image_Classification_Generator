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
        images = {}
        src_images = []
        full_data = []
        folder_count = 0
        data = os.listdir(self.path_data)
        if ".DS_Store" in data:  # Only necessary for MacOS
            os.remove(self.path_data + "/" + ".DS_Store")
            time.sleep(1)
            data = os.listdir(self.path_data)
        data = sorted_nicely(data)
        for folder in data:  # Iterate over each folder
            folder_count += 1
            f = os.listdir(self.path_data + "/" + folder)
            if ".DS_Store" in f:  # Only necessary for MacOS
                os.remove(self.path_data + "/" + folder + "/" + ".DS_Store")
                time.sleep(1)
                f = os.listdir(self.path_data + "/" + folder)
            file_count = 0
            for file in f:
                try:
                    full_data.append(file)
                    file_count += 1
                    image = cv2.imread(self.path_data + "/" + folder + "/" + file)  # Read the image in open-cv
                    src_img = plt.imread(self.path_data + "/" + folder + "/" + file)  # Read the same image in Matplotlib
                    image = cv2.resize(image, (dimx, dimy))  # Resize the open-cv image
                    # images.append(image)  # Append the resized image to a list
                    images["img " + str(folder_count) + "|" + str(file_count)] = [image, folder]  # Add the image and the folder (label) to a dict.
                    src_images.append(src_img)  # Append the source image to a list
                except Exception as e:
                    print("=" * 100)
                    print(e)
                    print("ERROR! OpenCV could not open the file '{}'\n"
                          "This is probably due to an invalid character in the file name or the file is corrupted in some way.\nRename "
                          "the file and try again or check if there are folders or non image files in the directory: "
                          "\n'{}'.".format(file, str(self.path_data) + folder))
                    print("=" * 100)
                    sys.exit(1)
            if len(f) == 0:
                print("=" * 100)
                print("ERROR! Your selected folder is empty: '{}'".format(str(self.path_data) + folder))
                print("=" * 100)
                sys.exit(1)
        return images, src_images, full_data

    # ============================================================
    def preprocess_data(self, images, shape):
        """ Preprocess the test data in the same way the training data was preprocessed """
        def normalize(img, img_normalize, channels):
            if channels == 3:
                pass
            else:
                img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Grayscale image
                # img = img[:, :, np.newaxis]  # Change img shape from (width, height) --> (width, height, 1)

            if img_normalize == "2":
                pass
            else:
                img = img / 255.0  # Normalize px values between 0 and 1
            return img

        img_normalize = str(self.df["img_normalize"][0])
        channels = shape[3]

        for file in images:
            images[file][0] = normalize(images[file][0], img_normalize, channels)  # Normalize and/or grayscale images

        x = []
        for file in images:
            x.append(images[file][0])
        x = np.array(x)  # TF needs numpy array
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1], channels)  # Reshape img to swap height and width

        for n, file in enumerate(images):
            images[file][0] = x[n]
        return images

    # ============================================================
    def predict_data(self, images, src_images, predictions, data, file, preview="y", show_instance="y", counter=1):
        def get_true_labels(df_labels):
            """ Map the real labels to the folder names"""
            label_names = df_labels[self.df["csv_column"][0]].tolist()  # Extract the labels and append it to a list
            folder_names = df_labels["folder_name"].tolist()  # Extract the labels and append it to a list
            label_names_str = []
            folder_names_str = []
            true_labels = []
            for n in range(len(label_names)):
                label_names_str.append(str(label_names[n]))  # Convert all labels to strings (Matplotlib displays only strings not numbers)
                folder_names_str.append(str(folder_names[n]).lower())  # Convert all folders to strings (Matplotlib displays only strings not numbers)

            for label in images:
                current_label = images[label][1]  # Get current folder name (label) of every image
                idx = folder_names_str.index(current_label.lower())  # Get index within the folder_names list
                true_labels.append(label_names_str[idx])  # Get the actual label to each image and append it to a new list
            return label_names_str, folder_names_str, true_labels

        def get_prediction_status(pred, real_label):
            """ Check if the prediction is correct and return True/False"""
            if pred.lower() == real_label.lower():  # Check if prediction is correct
                result = True
            else:
                result = False
            return result

        def get_accuracy(result_list):
            """ Calculate the accuracy based on the number of True's compared to the total entries in the array 'result_list'"""
            result_list = np.array(result_list)
            tp = np.count_nonzero(result_list)
            accuracy = round((tp / len(result_list))*100, 2)
            return accuracy

        if preview == "n":  # No preview images if u predict all models
            inp = "n"
        else:
            inp = input("Show preview images y/n (default = 'y'): ")  # Ask the user if preview images should be shown
        if self.gui == 0:
            labels_file_name = self.df["csv_name"][0]
            df_labels = pd.read_csv("labels/" + labels_file_name)  # Read the labels file from command line version
        else:
            labels_file_path = self.labels
            df_labels = pd.read_csv(labels_file_path + "/labels_generated.csv")  # Read the labels file from gui version


        label_names_str, folder_names_str, true_labels = get_true_labels(df_labels)  # Map the folder names to the real labels
        prediction_list = []
        result_list = []
        if inp == "n":  # Don't show preview images
            for i in range(len(src_images)):  # Iterate over all images
                pred = label_names_str[np.argmax(predictions[i])]  # Get the most likely prediction
                result = get_prediction_status(pred, true_labels[i])
                result_list.append(result)  # Append the result of the prediction to a list
                prediction_list.append(pred)  # Append the most likely prediction to a list
        else:  # Show preview images
            for i in range(len(src_images)):  # Iterate over all images
                pred = label_names_str[np.argmax(predictions[i])]  # Get the most likely prediction
                result = get_prediction_status(pred, true_labels[i])
                result_list.append(result)  # Append the result of the prediction to a list
                prediction_list.append(pred)  # Append the most likely prediction to a list
                image = src_images[i]
                plt.imshow(image)  # Show the source image
                plt.title("Prediction: " + pred + " --> " + str(result))  # Insert the most likely prediction
                plt.show()

        accuracy = get_accuracy(result_list)
        if show_instance == "n":
            self.df2[file] = prediction_list
            self.df2["Status " + str(counter)] = result_list
            self.acc_list.append(accuracy)
        else:
            df = {"File": data, "Prediction": prediction_list}  # Create a dictionary showing every file and the corresponding prediction
            df = pd.DataFrame(df)
            df["Status"] = result_list
            print("\nPredictions for model file:", file)
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))  # Print the dictionary as a nice table
            print("Accuracy on this data: {}%".format(accuracy))

    # ============================================================
    def initialize(self):
        def sorted_nicely(l):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        def get_img_list(images):
            """ Extract all images into one array and convert it to numpy array"""
            img_list = []
            for file in images:
                img_list.append(images[file][0])
            img_list = np.array(img_list)
            return img_list

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
                self.acc_list = []
                counter = 1
                for file in data:  # Iterate over each model
                    model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)  # load the model using the load_model function from keras
                    config = model.get_config()  # Get the config from the model
                    shape = config["layers"][0]["config"]["batch_input_shape"]  # Get the expected input shape
                    images, src_images, data_imgs = self.get_images(shape)
                    images = self.preprocess_data(images, shape)
                    img_list = get_img_list(images)
                    predictions = model.predict(img_list)
                    self.predict_data(images, src_images, predictions, data_imgs, file, "n", "n", counter)
                    counter += 1
                self.df2["File"] = data_imgs
                df = pd.DataFrame(self.df2)
                print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))  # Print the dictionary as a nice table
                best_idx = self.acc_list.index(max(self.acc_list))
                print("\n" + "="*100)
                for i in range(len(data)):
                    if i == best_idx:
                        print("'{}' achieved {}% accuracy on this data (best result)".format(data[i], self.acc_list[i]))
                    else:
                        print("'{}' achieved {}% accuracy on this data".format(data[i], self.acc_list[i]))
                print("=" * 100)
                sys.exit(1)  # Exit program after all models are predicted
        else:  # There is only one model file in the directory
            file = data[0]
            model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)  # load the model using the load_model function from keras

        config = model.get_config()  # Get the config from the model
        shape = config["layers"][0]["config"]["batch_input_shape"]  # Get the expected input shape
        images, src_images, data = self.get_images(shape)
        images = self.preprocess_data(images, shape)
        img_list = get_img_list(images)
        predictions = model.predict(img_list)
        self.predict_data(images, src_images, predictions, data, file)
