from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import platform
import tkinter
from tkintertable.Tables import TableCanvas
import os
import re
import pandas as pd
import webbrowser
import time
from python.preprocess import Preprocess
from python.augmentation import Augmentation
from python.predict import Predict
from python.predefined_model import Predefined_model
from PIL import ImageTk, Image

class GUI():
    def __init__(self, img=2):
        self.background_img = img
        self.settings = {}

        # Check which Operating system is used
        if platform.system() != "Windows":
            self.settings["os"] = "y"
            self.height = "560"
        else:
            self.settings["os"] = "n"
            self.height = "500"
        self.settings["mode"] = "1"


    def browse_testdirectory(self):
        # This funtion allows the user to select a folder from the pc
        filename = filedialog.askdirectory()
        self.predict_data.set(filename)


    def browse_trainingdirectory(self):
        # This funtion allows the user to select a folder from the pc
        filename = filedialog.askdirectory()
        self.training_path.set(filename)


    def browse_checkpointdirectory(self):
        # This funtion allows the user to select a folder from the pc
        filename = filedialog.askdirectory()
        self.checkpoint_path.set(filename)


    def browse_labelsdirectory(self):
        # This funtion allows the user to select a folder from the pc
        filename = filedialog.askdirectory()
        self.labels_path.set(filename)


    def clearcheckboxes(self):
        # This function clears all checkboxes in the augmentation options
        self.var1.set(0)
        self.var2.set(0)
        self.var3.set(0)
        self.var4.set(0)
        self.var5.set(0)
        self.var6.set(0)
        self.var7.set(0)
        self.var8.set(0)
        self.var9.set(0)
        self.var10.set(0)


    def delete_augmentations(self):
        augmentation = Augmentation(self.training_path.get())
        augmentation.delete_augmentations(self.settings, gui_mode=1)


    def sorted_nicely(self, l):
        # This function sorts the given iterable in the way that humans expect
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)


    def forwardTab(self, variable=2):
        # This function changes to the next tab/self.window
        self.tab_control.select(int(self.tab_control.index(self.tab_control.select())) + 1)
        if variable == 1:
            self.tab_control.select(int(self.tab_control.index(self.tab_control.select())) + 1)

        if int(self.tab_control.index(self.tab_control.select())) == 4:
            # If the tab id changes from 3 to 4 a table will be created showing all folders within the selected 'training_data' directory
            self.data = os.listdir(self.training_path.get())
            if ".DS_Store" in self.data:  # ONLY NECESSARY FOR MACOS
                os.remove(self.training_path.get() + "/" + ".DS_Store")
                time.sleep(0.8)
                self.data = os.listdir(self.training_path.get())
            if os.path.exists(self.training_path.get() + "/Insert your training data in this directory.txt"):
                os.remove(self.training_path.get() + "/Insert your training data in this directory.txt")
                time.sleep(0.8)
                self.data = os.listdir(self.training_path.get())
            self.data = self.sorted_nicely(self.data)

            table = TableCanvas(self.frame, rows=0, cols=0, cellwidth=250)
            table.createTableFrame()
            dictionary = {}
            labels_dir = os.listdir(self.labels_path.get())
            if "labels_generated.csv" in labels_dir:  # Check if there already exists a labels file
                label_df = pd.read_csv(self.labels_path.get() + "/labels_generated.csv")
                labels = label_df['label'].tolist()
                if len(labels) == len(self.data):  # Check if num(labels) = num(folders)
                    for n, i in enumerate(self.data):
                        dictionary[str(n)] = {'Folder_Name': i, 'Label': labels[n]}
                else:  # Labels = Folder_Names if num(labels) does not match num(folders)
                    for n, i in enumerate(self.data):
                        dictionary[str(n)] = {'Folder_Name': i, 'Label': i}
            else:  # Labels = Folder_Names if there is no labels file
                for n, i in enumerate(self.data):
                    dictionary[str(n)] = {'Folder_Name': i, 'Label': i}
            self.model = table.model
            self.model.importDict(dictionary)
            table.redraw()

        if int(self.tab_control.index(self.tab_control.select())) == 5:
            # If the tab id changes from 4 to 5 the table will
            # be saved as 'labels_generated.csv' in the selected 'labels' directory
            labels = []
            for i in range(len(self.data)):
                labels.append(str(self.model.getRecordAtRow(i)['Label']))
            df = {"label": labels, "folder_name": self.data}
            df = pd.DataFrame(df)
            file = "labels_generated.csv"
            df.to_csv(str(self.labels_path.get()) + "/" + file)  # save labels file as 'labels_generated.csv'
            self.settings["csv_name"] = file
            self.settings["csv_column"] = "label"

        if int(self.tab_control.index(self.tab_control.select())) == 9:
            # If the tab id changes from 8 to 9 the configuration of the model will be retrieved
            count_layers = 1
            pooling_layers = []
            if self.button2["text"] == "Disable Layer":
                count_layers += 1
            if self.button4["text"] == "Disable Layer":
                count_layers += 1
            if self.button6["text"] == "Disable Layer":
                count_layers += 1

            if self.button1["text"] == "Disable Layer":
                pooling_layers.append("y")
            else:
                if count_layers >= 1:
                    pooling_layers.append("n")
            if self.button3["text"] == "Disable Layer":
                pooling_layers.append("y")
            else:
                if count_layers >= 2:
                    pooling_layers.append("n")
            if self.button5["text"] == "Disable Layer":
                pooling_layers.append("y")
            else:
                if count_layers >= 3:
                    pooling_layers.append("n")
            if self.button7["text"] == "Disable Layer":
                pooling_layers.append("y")
            else:
                if count_layers >= 4:
                    pooling_layers.append("n")

            try:
                for i in range(self.settings["num_hidden_layers"]):
                    self.settings["hidden_layer_" + str(i + 1)] = str(self.neurons_in_hiddenlayer[i].get())
                    activation = str(self.activationfunction_in_hiddenlayer[i].get()).lower()
                    activationfunctions = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
                    if activation in activationfunctions:
                        self.settings["hidden_layer_activation_" + str(i + 1)] = activation
                    else:
                        self.settings["hidden_layer_activation_" + str(i + 1)] = "relu"
            except:
                self.settings["num_hidden_layers"] = 1
                self.settings["hidden_layer_1"] = ""
                self.settings["hidden_layer_activation_1"] = ""
            self.settings["count_layers"] = count_layers
            self.settings["pooling_layers"] = pooling_layers

            txtlist = [self.txt1.get(), self.txt2.get(), self.txt3.get(), self.txt4.get(), self.txt5.get(), self.txt6.get(), self.txt7.get(),
                       self.txt8.get(), self.txt9.get(), self.txt10.get(), self.txt11.get(), self.txt12.get(), self.txt13.get(), self.txt14.get(),
                       self.txt15.get(), self.txt16.get(), self.txt17.get(), self.txt18.get(), self.txt19.get(), self.txt20.get()]

            count = 0
            counter = 1
            for i in range(count_layers):
                self.settings["num_neurons_" + str(i + 1)] = str(txtlist[i + count])
                self.settings["strides_neurons_" + str(i + 1)] = str(txtlist[i + 1 + count]) + " " + str(txtlist[i + 2 + count])
                if pooling_layers[i] == "y":
                    self.settings["max_pool_" + str(i + 1)] = str(txtlist[i + 3 + count]) + " " + str(txtlist[i + 4 + count])
                if counter == 1:
                    self.settings["activation_type_1"] = str(self.firstconv2d).lower()
                if counter == 2:
                    self.settings["activation_type_2"] = str(self.secondconv2d).lower()
                if counter == 3:
                    self.settings["activation_type_3"] = str(self.thirdconv2d).lower()
                if counter == 4:
                    self.settings["activation_type_4"] = str(self.fourthconv2d).lower()
                count += 4
                counter = counter + 1


    def selectoption_train_test(self, variable):
        # Changes Tab either to select option, train model or predict images
        if variable == 0:
            self.tab_control.select(0)
        if variable == 1:
            self.tab_control.select(1)
        if variable == 2:
            self.tab_control.select(2)


    def backwardTab(self, variable=2):
        # This function changes the tab id to the previous tab id
        self.tab_control.select(int(self.tab_control.index(self.tab_control.select())) - 1)
        if variable == 1:
            self.tab_control.select(int(self.tab_control.index(self.tab_control.select())) - 1)


    def hide_widget(self, widget):
        # This function hides a widget so that the user cant see it / interact with it
        widget.pack_forget()


    def show_widget(self, widget):
        # This function shows a hidden widget again
        widget.pack()


    def openweb(self, url):
        # Opens a given url in the standard browser
        webbrowser.open(url, new=1)


    def customize_hiddenlayers(self, variable, counter):
        # Adds hiddenlayers and dropout to model in model configuration
        secondcounter = 1
        self.settings["num_hidden_layers"] = int(variable)
        for i in range(variable):
            standardvalue_hiddenlayer_neurons = StringVar()
            standardvalue_hiddenlayer_neurons.set('64')

            standardactivationfunction_hiddenlayer = StringVar()
            standardactivationfunction_hiddenlayer.set('Relu')

            lb = Label(self.frame5, text="Choose number of neurons for hidden layer '" + str(i + 1) + "' and activation function: ")

            lb.grid(column=0, row=counter)

            txt = Entry(self.frame5, textvariable=standardvalue_hiddenlayer_neurons, width=10)

            txt.grid(column=1, row=counter)

            txt = Entry(self.frame5, textvariable=standardactivationfunction_hiddenlayer, width=10)

            txt.grid(column=2, row=counter)
            if counter == 27:
                self.button50 = Button(self.frame5, text="More information about activation functions",
                                  command=lambda: self.openweb("https://keras.io/api/layers/activations/"))
                self.button50.grid(column=3, row=counter)

            counter = counter + 1
            secondcounter = secondcounter + 1
            self.neurons_in_hiddenlayer.append(standardvalue_hiddenlayer_neurons)
            self.activationfunction_in_hiddenlayer.append(standardactivationfunction_hiddenlayer)

        # Add second dropout layer
        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=counter + 1, sticky="ew", pady=(15, 15))

        standardvalue_hiddenlayers = StringVar()
        standardvalue_hiddenlayers.set('1')

        lb = Label(self.frame5, text="Choose dropout ratio in %: ")

        lb.grid(column=0, row=counter + 2, pady=(15, 0))

        self.standardvalue_dropoutratio2 = StringVar()
        self.standardvalue_dropoutratio2.set('25')

        self.txt23 = Entry(self.frame5, textvariable=self.standardvalue_dropoutratio2, state=DISABLED, width=10)
        self.txt23.grid(column=1, row=counter + 2, pady=(15, 0))

        lb = Label(self.frame5, text="")

        lb.grid(column=2, row=counter + 2, pady=(15, 0))

        self.button10 = ttk.Button(self.frame5, text="Use Layer", command=lambda: self.change_entry_state(self.button10, [self.txt23]))
        self.button10.grid(column=3, row=counter + 2)


    def change_entry_state(self, button, entry_to_enable):
        # Enables entry for layer if button "Use Layer" is selected and changes button text to "Disable Layer"
        # If the button text is "Disable Layer", disable entry and change button text to "Use Layer"
        # Checks if Conv2d Layers are enabled. If a Conv2d Layer is not enabled the respective Maxpooling Layer will not activate

        if self.txt6["state"] == 'disabled':
            allowpool2 = 0
        if self.txt6["state"] == 'normal':
            allowpool2 = 1
        if self.txt11["state"] == 'disabled':
            allowpool3 = 0
        if self.txt11["state"] == 'normal':
            allowpool3 = 1
        if self.txt16["state"] == 'disabled':
            allowpool4 = 0
        if self.txt16["state"] == 'normal':
            allowpool4 = 1

        if entry_to_enable[0] == self.txt9 and allowpool2 == 1 or entry_to_enable[0] == self.txt14 and allowpool3 == 1 or entry_to_enable[
            0] == self.txt19 and allowpool4 == 1:
            if button['text'] == "Use Layer":
                button.config(text="Disable Layer")
                for i in entry_to_enable:
                    i.configure(state="normal")
            else:
                button.config(text="Use Layer")
                for i in entry_to_enable:
                    i.configure(state="disabled")
        else:
            if entry_to_enable[0] == self.txt9 or entry_to_enable[0] == self.txt14 or entry_to_enable[0] == self.txt19:
                pass
            else:
                if button['text'] == "Use Layer":
                    button.config(text="Disable Layer")
                    for i in entry_to_enable:
                        i.configure(state="normal")
                else:
                    button.config(text="Use Layer")
                    for i in entry_to_enable:
                        if i == self.txt6:
                            self.txt9.configure(state="disabled")
                            self.txt10.configure(state="disabled")
                            self.txt11.configure(state="disabled")
                            self.txt12.configure(state="disabled")
                            self.txt13.configure(state="disabled")
                            self.txt14.configure(state="disabled")
                            self.txt15.configure(state="disabled")
                            self.txt16.configure(state="disabled")
                            self.txt17.configure(state="disabled")
                            self.txt18.configure(state="disabled")
                            self.txt19.configure(state="disabled")
                            self.txt20.configure(state="disabled")
                            self.button3.config(text="Use Layer")
                            self.button4.config(text="Use Layer", state="disabled")
                            self.button5.config(text="Use Layer", state="disabled")
                            self.button6.config(text="Use Layer", state="disabled")
                            self.button7.config(text="Use Layer", state="disabled")

                        if i == self.txt11:
                            self.txt14.configure(state="disabled")
                            self.txt15.configure(state="disabled")
                            self.txt16.configure(state="disabled")
                            self.txt17.configure(state="disabled")
                            self.txt18.configure(state="disabled")
                            self.txt19.configure(state="disabled")
                            self.txt20.configure(state="disabled")
                            self.button5.config(text="Use Layer")
                            self.button6.config(text="Use Layer", state="disabled")
                            self.button7.config(text="Use Layer", state="disabled")
                        if i == self.txt16:
                            self.txt19.configure(state="disabled")
                            self.txt20.configure(state="disabled")
                            self.button7.config(text="Use Layer")
                        i.configure(state="disabled")


    def reset_scrollregion(self, event):
        # Allows that scroll region is adaptive
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def on_mousewheel(self, event):
        # Allows scrolling with mousewheel
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def callback_delete(self, selection):
        # Updates settings for selected option in respective OptionMenu
        if selection == "Don't delete any previous model files":
            self.settings["model_save"] = "1"
        if selection == "Delete oldest existing model file":
            self.settings["model_save"] = "2"
        if selection == "Delete all existing model files":
            self.settings["model_save"] = "3"


    def callback_firstconv2d(self, selection):
        # Updates settings for selected option in respective OptionMenu
        self.firstconv2d = selection


    def callback_secondconv2d(self, selection):
        # Updates settings for selected option in respective OptionMenu
        self.secondconv2d = selection


    def callback_thirdconv2d(self, selection):
        # Updates settings for selected option in respective OptionMenu
        self.thirdconv2d = selection


    def callback_fourthconv2d(self, selection):
        # Updates settings for selected option in respective OptionMenu
        self.fourthconv2d = selection


    def callback_excution_options(self, selection):
        # Updates settings for selected option in respective OptionMenu
        if selection == "Automatic":
            self.settings["mode"] = "1"
        if selection == "Use GPU for training and CPU for predicting with memory growth enabled for the GPU":
            self.settings["mode"] = "2"
        if selection == "Use GPU for training and predicting":
            self.settings["mode"] = "3"
        if selection == "Force CPU for training and predicting":
            self.settings["mode"] = "4"


    def updatesettings(self):
        # Retrieve augmentation options and model architecture settings and update the respective settings
        augmentation_inp = ""
        if self.var1.get() == 1: augmentation_inp += "2 "
        if self.var9.get() == 1: augmentation_inp += "3 "
        if self.var3.get() == 1: augmentation_inp += "4 "
        if self.var4.get() == 1: augmentation_inp += "5 "
        if self.var5.get() == 1: augmentation_inp += "6 "
        if self.var6.get() == 1: augmentation_inp += "7 "
        if self.var7.get() == 1: augmentation_inp += "8 "
        if self.var8.get() == 1: augmentation_inp += "9 "
        if self.var10.get() == 1: augmentation_inp += "10"
        augmentation_inp = augmentation_inp.split()


        self.settings["dropout_1"] = ""
        self.settings["dropout_2"] = ""
        self.settings["dim"] = str(self.standardvalue_width.get()) + " " + str(self.standardvalue_height.get())
        if self.gss[0].instate(['selected']):
            self.settings["channels"] = 1
        else:
            self.settings["channels"] = 2
        if self.gs[0].instate(['selected']):
            self.settings["normalize"] = 1
        else:
            self.settings["normalize"] = 2
        if int(self.standardvalue_validationsize.get()) < 1:
            self.standardvalue_validationsize.set(20)
            print("Changed validationsize to 20 because of invalid input")
        if int(self.standardvalue_validationsize.get()) > 99:
            self.standardvalue_validationsize.set(20)
            print("Changed validationsize to 20 because of invalid input")
        if int(self.standardvalue_epochs.get()) < 1:
            self.standardvalue_epochs.set(10)
            print("Changed epochs to 10 because of invalid input")
        if int(self.standardvalue_batchsize.get()) < 1:
            self.standardvalue_batchsize.set(64)
            print("Changed batchsize to 64 because of invalid input")
        self.settings["validation"] = int(self.standardvalue_validationsize.get())
        self.settings["epochs"] = int(self.standardvalue_epochs.get())
        self.settings["batch_size"] = int(self.standardvalue_batchsize.get())

        if self.txt21["state"] == 'normal':
            dropout = int(self.standardvalue_dropoutratio1.get())
            if dropout > 99:
                print("Changed dropout rate for the first dropout layer to 25 % because of invalid input")
                dropout = 25
            if dropout < 0:
                dropout = 25
                print("Changed dropout rate for the first dropout layer to 25 % because of invalid input")
            self.settings["dropout_1"] = str(dropout)

        try:
            if self.txt23["state"] == 'normal':
                dropout = int(self.standardvalue_dropoutratio2.get())
                if dropout > 99:
                    print("Changed dropout rate for the second dropout layer to 25 % because of invalid input")
                    dropout = 25
                if dropout < 0:
                    print("Changed dropout rate for the second dropout layer to 25 % because of invalid input")
                    dropout = 25
                self.settings["dropout_2"] = str(dropout)
        except:
            pass

        if self.settings["predefined_model"] == "y":  # Use configuration of the predefined model
            predefined_model = Predefined_model()
            self.settings = predefined_model.initialize(self.settings)

        self.settings["s_time"] = time.time()
        self.window.destroy()
        ###################################################################################################
        # Start the pipeline
        ###################################################################################################
        if augmentation_inp:
            augmentation = Augmentation(self.training_path.get())
            augmentation.initialize(augmentation_inp, self.settings)
        preprocess = Preprocess(self.training_path.get(), self.checkpoint_path.get())
        preprocess.initialize(self.settings)

    def predict_images(self):
        self.window.destroy()
        predict = Predict(self.predict_data.get(), self.checkpoint_path.get(), self.labels_path.get(), 1)
        predict.initialize()

    # ******************** Help Buttons Info *****************
    def tab1_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x300")

        # A Label widget to show in toplevel
        Label(newWindow, text="Training data directory: The directory where the training data is stored").pack(pady=(15, 0))

        # A Label widget to show in toplevel
        Label(newWindow, text="Checkpoint directory: The directory where the checkpoints will be stored ").pack(pady=(15, 0))
        Label(newWindow, text="Labels directory: The directory where the labels will be stored ").pack(pady=(15, 0))


    def tab2_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("700x700")

        # A Label widget to show in toplevel
        Label(newWindow,
              text="Image augmentation is a technique that increases the amount of training images by changing some aspects of the image and saving it as a separate file. You have various options for image augmentation:",
              wraplengt=500).pack(pady=(15, 0))

        # A Label widget to show in toplevel
        Label(newWindow, text="90 degrees rotation --> Rotate an image by 90 degrees to the right").pack(pady=(15, 0))
        Label(newWindow, text="180 degrees rotation --> Rotate an image by 180 degrees").pack(pady=(15, 0))
        Label(newWindow, text="270 degrees rotation --> Rotate an image by 270 degrees to the right").pack(pady=(15, 0))
        Label(newWindow, text="Randomly flip image to the left or right --> Image will randomly be flipped to the left or right").pack(pady=(15, 0))
        Label(newWindow, text="Randomly flip up or down --> Image will randomly be flipped up or down").pack(pady=(15, 0))
        Label(newWindow, text="Randomly change hue --> The hue of the image will randomly be changed based on a delta of 0.1").pack(pady=(15, 0))
        Label(newWindow, text="Randomly change saturation --> The saturation of the image will randomly be changed between 0.6 and 1.6").pack(
            pady=(15, 0))
        Label(newWindow, text="Randomly change brightness --> The brightness of an image will randomly be changed based on a delta of 0.05").pack(
            pady=(15, 0))
        Label(newWindow, text="Randomly change contrast --> The contrast will randomly be changed between 0.7 and 1.3").pack(pady=(15, 0))


    def tab3_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x300")

        # A Label widget to show in toplevel
        Label(newWindow, text="A label file is used for assigning your folders labels. You can provide an own name for each folder in the Label column",
              wraplengt=300).pack(pady=(15, 0))


    def tab4_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x300")

        # A Label widget to show in toplevel
        Label(newWindow, text="The preprocessing transforms the data in a way that TF accepts the data as a valid input.", wraplengt=500).pack(
            pady=(15, 0))
        Label(newWindow,
              text="Resize all images to a specific shape: All images need to be the same shape before they can be passed to the model. Type e.g. '64 32' to resize all images to 64px width and 32px height. Alternatively you can just press to enter to the resize all images to the shape of the first image found.",
              wraplengt=500).pack(pady=(15, 0))
        Label(newWindow,
              text="Grayscale images: You can choose if you want to transform all images to black and white by typing '1' or pressing enter. By typing '2' all images will be unchanged.",
              wraplengt=500).pack(pady=(15, 0))
        Label(newWindow,
              text="Normalization of images: Normalize the pixel values of all images by typing '1' or pressing enter. The pixel value of normalized images only range from 0 to 1 instead of 0 to 255.",
              wraplengt=500).pack(pady=(15, 0))


    def tab5_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x300")

        # A Label widget to show in toplevel
        Label(newWindow, text="In training options you can adjust 3 major settings:", wraplengt=500).pack(pady=(15, 0))
        Label(newWindow,
              text="Validation size: Validation size specifies how much % of a data is used to test the model accuracy and loss. Type e.g. '20' for a validation size of 20 %.",
              wraplengt=500).pack(pady=(15, 0))
        Label(newWindow,
              text="Number of epochs: Choose how often you want to iterate over all training images. If this number is to small your model will underfit but if the number is too high on the other hand your model will likely overfit",
              wraplengt=500).pack(pady=(15, 0))
        Label(newWindow,
              text="Batch size: The batch size specifies how many images you pass to the model at once before the model weights are updated. A higher batch size can increase the training speed as well as the model quality at the cost of more ram. Keep in mind that a batch size that is too high can lead to worse generalization in some cases. This means that the highest possible batch size is not necessarily the best option. You can read more about batch size in this article.",
              wraplengt=500).pack(pady=(15, 0))


    def tab6_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x300")

        # A Label widget to show in toplevel
        Label(newWindow,
              text="You can either define the model structure yourself or use the predefined model structure. The model structure of the chosen model will be saved in the file model_summary.txt when the training starts. The structure of the predefined model can be viewed in the file predefined_model_summary.txt.",
              wraplengt=300).pack(pady=(15, 0))


    def tab7_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x300")

        # A Label widget to show in toplevel
        Label(newWindow,
              text="When defining your own model structure you can add up to four convolutional and max pooling layers, a dropout layer before the first hidden layer, any desired amount of hidden layers, and a dropout layer before the output layer.",
              wraplengt=300).pack(pady=(15, 0))
        Label(newWindow, text="You can specify the number of neurons, the strides and the activation function for the convolutional layers.",
              wraplengt=300).pack(pady=(15, 0))
        Label(newWindow, text="You can specify the pool size for the MaxPooling layers.", wraplengt=300).pack(pady=(15, 0))
        Label(newWindow, text="You can specify the dropout rate (in %) for the dropout layers.", wraplengt=300).pack(pady=(15, 0))
        Label(newWindow, text="You can specify the number of neurons as well as the activation function for the hidden layers.", wraplengt=300).pack(
            pady=(15, 0))


    def tab8_help(self):
        # Creates a new self.window with helping information for the respective tab
        # Toplevel object which will
        # be treated as a new self.window
        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
        newWindow.title("More Information")

        # sets the geometry of toplevel
        newWindow.geometry("500x500")

        Label(newWindow,
              text="Depending on your installation you might want to choose the execution mode yourself. That's why you have four different options available. ",
              wraplengt=300).pack(pady=(15, 0))
        Label(newWindow,
              text="Automatic = Do not specify any execution mode. TF will choose a mode based on your installation automatically. (If you have an NVIDIA GPU and CUDA installed, and you get a TF error by running on automatic mode try one of the other options)",
              wraplengt=300).pack(pady=(15, 0))
        Label(newWindow,
              text="GPU for training and CPU for predicting = Enables memory growth for GPU which should eliminate most errors with the GPU version of TF. CPU is used for predicting images",
              wraplengt=300).pack(pady=(15, 0))
        Label(newWindow,
              text="GPU for training and predicting = Same as option [2] but also uses GPU for predicting. (Note: Predicting needs very little performance and is pretty fast most of the time, but the initialization of CUDA takes a few seconds. That is why option [2] is the better execution mode in most situations.)",
              wraplengt=300).pack(pady=(15, 0))
        Label(newWindow, text="Force CPU for training and predicting = If you have any problems whatsoever with you GPU, use this mode.",
              wraplengt=300).pack(pady=(15, 0))


    def changevariable(self, i):
        variable = i
        self.forwardTab(variable)
        if i == 0:
            self.settings["predefined_model"] = "n"
        else:
            self.settings["predefined_model"] = "y"


    def resize_image(self, event):
        new_width = event.width
        new_height = event.height
        self.image = self.img_copy.resize((new_width, new_height))
        self.background_image1 = ImageTk.PhotoImage(self.image)
        self.background1.configure(image=self.background_image1)
        self.background2.configure(image=self.background_image1)
        self.background3.configure(image=self.background_image1)
        self.background4.configure(image=self.background_image1)
        self.background5.configure(image=self.background_image1)
        self.background6.configure(image=self.background_image1)
        self.background7.configure(image=self.background_image1)
        self.background8.configure(image=self.background_image1)
        self.background9.configure(image=self.background_image1)
        self.background10.configure(image=self.background_image1)


    def buildgui(self):
        ###################################################################################################
        # Build the GUI
        ###################################################################################################
        # Saves the layer configuration for the hiddenlayers in model configuration
        self.neurons_in_hiddenlayer = []
        self.activationfunction_in_hiddenlayer = []

        # Initialize GUI with a 800xself.height self.window
        self.window = Tk()

        self.window.geometry('800x'+self.height)
        self.window.title("Automated Image Classifier")

        # Hide tabs from user
        style = ttk.Style()
        style.layout('TNotebook.Tab', [])

        # Creates tabs for the GUI to navigate through the different options for the application
        # In addition sets a title for every tab
        self.tab_control = ttk.Notebook(self.window)
        self.image = Image.open("python/gui_background_img/gui_background_img" + str(self.background_img) + ".png")
        self.img_copy = self.image.copy()
        self.background_image1 = ImageTk.PhotoImage(self.image)


        tab1 = ttk.Frame(self.tab_control)
        lb = Label(tab1, text='Path Settings')
        lb.pack(pady=(0, 15), padx=(10, 100))

        tab2 = ttk.Frame(self.tab_control)
        lb = Label(tab2, text='Augmentation Options')
        lb.pack(pady=(0, 15), padx=(10, 100))

        tab3 = ttk.Frame(self.tab_control)
        lb = Label(tab3, text='Label Options')
        lb.pack(pady=(0, 15), padx=(10, 100))

        tab4 = ttk.Frame(self.tab_control)
        lb = Label(tab4, text='Preprocess Options')
        lb.pack(pady=(0, 15))

        tab5 = ttk.Frame(self.tab_control)
        lb = Label(tab5, text='Training Options')
        lb.pack(pady=(0, 15))

        tab6 = ttk.Frame(self.tab_control)
        lb = Label(tab6, text='Model Options')
        lb.pack(pady=(0, 15))

        tab7 = ttk.Frame(self.tab_control)
        lb = Label(tab7, text='Model Configuration')
        lb.pack(pady=(0, 15))

        tab8 = ttk.Frame(self.tab_control)
        lb = Label(tab8, text='Execution Options')
        lb.pack(pady=(0, 15))

        tab9 = ttk.Frame(self.tab_control)
        lb = Label(tab9, text='Selection Option')
        lb.pack(pady=(0, 15))

        tab10 = ttk.Frame(self.tab_control)
        lb = Label(tab10, text='Predict')
        lb.pack(pady=(0, 15))

        self.background1 = Label(tab1, image=self.background_image1)
        self.background1.place(x=0, y=0, relwidth=1, relheight=1)
        self.background1.bind('<Configure>', self.resize_image)

        self.background2 = Label(tab2, image=self.background_image1)
        self.background2.place(x=0, y=0, relwidth=1, relheight=1)
        self.background2.bind('<Configure>', self.resize_image)

        self.background3 = Label(tab3, image=self.background_image1)
        self.background3.place(x=0, y=0, relwidth=1, relheight=1)
        self.background3.bind('<Configure>', self.resize_image)

        self.background4 = Label(tab4, image=self.background_image1)
        self.background4.place(x=0, y=0, relwidth=1, relheight=1)
        self.background4.bind('<Configure>', self.resize_image)

        self.background5 = Label(tab5, image=self.background_image1)
        self.background5.place(x=0, y=0, relwidth=1, relheight=1)
        self.background5.bind('<Configure>', self.resize_image)

        self.background6 = Label(tab6, image=self.background_image1)
        self.background6.place(x=0, y=0, relwidth=1, relheight=1)
        self.background6.bind('<Configure>', self.resize_image)

        self.background7 = Label(tab7, image=self.background_image1)
        self.background7.place(x=0, y=0, relwidth=1, relheight=1)
        self.background7.bind('<Configure>', self.resize_image)

        self.background8 = Label(tab8, image=self.background_image1)
        self.background8.place(x=0, y=0, relwidth=1, relheight=1)
        self.background8.bind('<Configure>', self.resize_image)

        self.background9 = Label(tab9, image=self.background_image1)
        self.background9.place(x=0, y=0, relwidth=1, relheight=1)
        self.background9.bind('<Configure>', self.resize_image)

        self.background10 = Label(tab10, image=self.background_image1)
        self.background10.place(x=0, y=0, relwidth=1, relheight=1)
        self.background10.bind('<Configure>', self.resize_image)

        self.tab_control.add(tab9, text='Selection Option')
        self.tab_control.add(tab10, text='Predict')
        self.tab_control.add(tab1, text='Path Settings')
        self.tab_control.add(tab2, text='Augmentation Options')
        self.tab_control.add(tab3, text='Label Options')
        self.tab_control.add(tab4, text='Preprocess Options')
        self.tab_control.add(tab5, text='Training Options')
        self.tab_control.add(tab6, text='Model Options')
        self.tab_control.add(tab7, text='Model Configuration')
        self.tab_control.add(tab8, text='Execution Options')


        # ******* Used Variables for the Application *************
        # Sets standard values for activation functions in model configuration
        self.firstconv2d = "relu"
        self.secondconv2d = "relu"
        self.thirdconv2d = "relu"
        self.fourthconv2d = "relu"
        self.settings["model_save"] = "1"

        # Saves the configured paths
        self.predict_data = StringVar()
        self.predict_data.set('predict_data')

        self.training_path = StringVar()
        self.training_path.set('training_data')

        self.checkpoint_path = StringVar()
        self.checkpoint_path.set('checkpoints')

        self.labels_path = StringVar()
        self.labels_path.set('labels')

        # Saves selected augmentation options
        self.var1 = IntVar()
        self.var2 = IntVar()
        self.var3 = IntVar()
        self.var4 = IntVar()
        self.var5 = IntVar()
        self.var6 = IntVar()
        self.var7 = IntVar()
        self.var8 = IntVar()
        self.var9 = IntVar()
        self.var10 = IntVar()

        # Saves input dimensions for neural network
        self.standardvalue_width = StringVar()
        self.standardvalue_width.set('64')

        self.standardvalue_height = StringVar()
        self.standardvalue_height.set('32')

        # Saves selected preprocess options
        var = tkinter.IntVar()
        self.var2 = tkinter.IntVar()

        # Saves training options
        self.standardvalue_validationsize = IntVar()
        self.standardvalue_validationsize.set(20)

        self.standardvalue_epochs = IntVar()
        self.standardvalue_epochs.set(10)

        self.standardvalue_batchsize = IntVar()
        self.standardvalue_batchsize.set(64)

        # Saves option for previous model files
        clicked = StringVar()
        clicked.set("Don't delete any previous model files")

        # Saves the custom model configuration
        standardvalue_conv2d = StringVar()
        standardvalue_conv2d.set('32')

        standardvalue_conv2d_stride_one = StringVar()
        standardvalue_conv2d_stride_one.set('3')

        standardvalue_conv2d_stride_two = StringVar()
        standardvalue_conv2d_stride_two.set('3')

        clicked2 = StringVar()
        clicked2.set("Relu")

        standardvalue_maxpooling_stride_one = StringVar()
        standardvalue_maxpooling_stride_one.set('2')

        standardvalue_maxpooling_stride_two = StringVar()
        standardvalue_maxpooling_stride_two.set('2')

        standardvalue_conv2d2 = StringVar()
        standardvalue_conv2d2.set('32')

        standardvalue_conv2d_stride_one2 = StringVar()
        standardvalue_conv2d_stride_one2.set('3')

        standardvalue_conv2d_stride_two2 = StringVar()
        standardvalue_conv2d_stride_two2.set('3')

        clicked3 = StringVar()
        clicked3.set("Relu")

        standardvalue_maxpooling_stride_one2 = StringVar()
        standardvalue_maxpooling_stride_one2.set('2')

        standardvalue_maxpooling_stride_two2 = StringVar()
        standardvalue_maxpooling_stride_two2.set('2')

        standardvalue_conv2d3 = StringVar()
        standardvalue_conv2d3.set('32')

        standardvalue_conv2d_stride_one3 = StringVar()
        standardvalue_conv2d_stride_one3.set('3')

        standardvalue_conv2d_stride_two3 = StringVar()
        standardvalue_conv2d_stride_two3.set('3')

        clicked4 = StringVar()
        clicked4.set("Relu")

        standardvalue_maxpooling_stride_one3 = StringVar()
        standardvalue_maxpooling_stride_one3.set('2')

        standardvalue_maxpooling_stride_two3 = StringVar()
        standardvalue_maxpooling_stride_two3.set('2')

        standardvalue_conv2d4 = StringVar()
        standardvalue_conv2d4.set('32')

        standardvalue_conv2d_stride_one4 = StringVar()
        standardvalue_conv2d_stride_one4.set('3')

        standardvalue_conv2d_stride_two4 = StringVar()
        standardvalue_conv2d_stride_two4.set('3')

        clicked5 = StringVar()
        clicked5.set("Relu")

        standardvalue_maxpooling_stride_one4 = StringVar()
        standardvalue_maxpooling_stride_one4.set('2')

        standardvalue_maxpooling_stride_two4 = StringVar()
        standardvalue_maxpooling_stride_two4.set('2')

        self.standardvalue_dropoutratio1 = StringVar()
        self.standardvalue_dropoutratio1.set('25')

        standardvalue_hiddenlayers = StringVar()
        standardvalue_hiddenlayers.set('1')

        # Save selected execution option
        clicked6 = StringVar()
        clicked6.set("Automatic")


        # *********************** Help Buttons *****************
        # Create buttons to access helping information for respective tab
        help_button = ttk.Button(tab1, text="Help", command=lambda: self.tab1_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab2, text="Help", command=lambda: self.tab2_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab3, text="Help", command=lambda: self.tab3_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab4, text="Help", command=lambda: self.tab4_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab5, text="Help", command=lambda: self.tab5_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab6, text="Help", command=lambda: self.tab6_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab7, text="Help", command=lambda: self.tab7_help())
        help_button.pack(side=BOTTOM)

        help_button = ttk.Button(tab8, text="Help", command=lambda: self.tab8_help())
        help_button.pack(side=BOTTOM)

        # *********************** Change Tabs *****************
        # Create buttons to navigate through the GUI
        forwardButton = ttk.Button(tab1, text="-->", command=lambda: self.forwardTab()).pack(side='right', anchor='s')

        backwardButton = ttk.Button(tab1, text="<--", command=lambda: self.selectoption_train_test(0)).pack(side='bottom', anchor='w')

        backwardButton = ttk.Button(tab10, text="<--", command=lambda: self.selectoption_train_test(0)).pack(side='bottom', anchor='w')

        forwardButton = ttk.Button(tab2, text="-->", command=lambda: self.forwardTab()).pack(side='right', anchor='s')

        backwardButton = ttk.Button(tab2, text="<--", command=lambda: self.backwardTab()).pack(side='bottom', anchor='w')

        forwardButton = ttk.Button(tab3, text="-->", command=lambda: self.forwardTab()).pack(side='right', anchor='s')

        backwardButton = ttk.Button(tab3, text="<--", command=lambda: self.backwardTab()).pack(side='bottom', anchor='w')

        forwardButton = ttk.Button(tab4, text="-->", command=lambda: self.forwardTab()).pack(side='right', anchor='s')

        backwardButton = ttk.Button(tab4, text="<--", command=lambda: self.backwardTab()).pack(side='bottom', anchor='w')

        forwardButton = ttk.Button(tab5, text="-->", command=lambda: self.forwardTab()).pack(side='right', anchor='s')

        backwardButton = ttk.Button(tab5, text="<--", command=lambda: self.backwardTab()).pack(side='bottom', anchor='w')

        backwardButton = ttk.Button(tab6, text="<--", command=lambda: self.backwardTab()).pack(side='bottom', anchor='w')

        forwardButton = ttk.Button(tab7, text="-->", command=lambda: self.forwardTab()).pack(side='right', anchor='s')

        backwardButton = ttk.Button(tab7, text="<--", command=lambda: self.backwardTab()).pack(side='bottom', anchor='w')

        backwardButton = ttk.Button(tab8, text="<--", command=lambda: self.backwardTab(1)).pack(side='bottom', anchor='w')


        # *********************** Select option *****************
        # Create buttons to select whether user wants to train model or predict images with the application
        btn = ttk.Button(tab9, text="Train model", width=20, command=lambda: self.selectoption_train_test(2))
        btn.pack(pady=(150, 0))

        btn = ttk.Button(tab9, text="Test model", width=20, command=lambda: self.selectoption_train_test(1))
        btn.pack(pady=(15, 0))

        # *********************** Test model *****************
        # Allows user to select path for testdata and predict images in selected directory
        label = Label(tab10, text="Select paths: ", padx=5, pady=10)
        label.pack(padx=(60, 0))
        lbl = Label(tab10, width=20, text="Test data directory")
        lbl.pack(pady=(30, 0), padx=(60, 0))
        txt = Entry(tab10, textvariable=self.predict_data, state=DISABLED, width=50)
        txt.pack(pady=(5, 0), padx=(60, 0))

        button2 = ttk.Button(tab10, text="Browse", command=lambda: self.browse_testdirectory())
        button2.pack(pady=(5, 0), padx=(60, 0))
        lb2 = Label(tab10, width=20, text="Checkpoint directory")
        lb2.pack(pady=(20, 0), padx=(60, 0))
        txt = Entry(tab10, textvariable=self.checkpoint_path, state=DISABLED, width=50)
        txt.pack(pady=(5, 0), padx=(60, 0))

        button2 = ttk.Button(tab10, text="Browse", command=lambda: self.browse_checkpointdirectory())
        button2.pack(pady=(5, 0), padx=(60, 0))
        lb3 = Label(tab10, width=20, text="Labels directory")
        lb3.pack(pady=(20, 0), padx=(60, 0))
        txt = Entry(tab10, textvariable=self.labels_path, state=DISABLED, width=50)
        txt.pack(pady=(5, 0), padx=(60, 0))

        button3 = ttk.Button(tab10, text="Browse", command=lambda: self.browse_labelsdirectory())
        button3.pack(pady=(5, 0), padx=(60, 0))

        predict_button = ttk.Button(tab10, width=20, text="Predict Images", command=lambda: self.predict_images())
        predict_button.pack(pady=(35, 0), padx=(60, 0))

        # *********************** TAB 1 *****************
        # Allows users to configure the necessary paths for the application
        label = Label(tab1, text="Select paths: ", padx=5, pady=10)
        label.pack(padx=(60, 0))
        lbl = Label(tab1, width=20, text="Training data directory")
        lbl.pack(pady=(30, 0), padx=(60, 0))
        txt = Entry(tab1, textvariable=self.training_path, state=DISABLED, width=50)
        txt.pack(pady=(5, 0), padx=(60, 0))

        self.button2 = ttk.Button(tab1, text="Browse", command=lambda: self.browse_trainingdirectory())
        self.button2.pack(pady=(5, 0), padx=(60, 0))
        lb2 = Label(tab1, width=20, text="Checkpoint directory")
        lb2.pack(pady=(40, 0), padx=(60, 0))
        txt = Entry(tab1, textvariable=self.checkpoint_path, state=DISABLED, width=50)
        txt.pack(pady=(5, 0), padx=(60, 0))

        self.button2 = ttk.Button(tab1, text="Browse", command=lambda: self.browse_checkpointdirectory())
        self.button2.pack(pady=(5, 0), padx=(60, 0))
        lb3 = Label(tab1, width=20, text="Labels directory")
        lb3.pack(pady=(40, 0), padx=(60, 0))
        txt = Entry(tab1, textvariable=self.labels_path, state=DISABLED, width=50)
        txt.pack(pady=(5, 0), padx=(60, 0))

        self.button3 = ttk.Button(tab1, text="Browse", command=lambda: self.browse_labelsdirectory())
        self.button3.pack(pady=(5, 0), padx=(60, 0))


        # *********************** TAB 2 *****************
        # Allows users to configure the wanted augmentation
        label = Label(tab2, text="Select Augmentation Options below: ", padx=5, pady=10)
        label.pack(padx=(60, 0))

        ttk.Checkbutton(tab2, text="Rotate by 90 degrees", width=27, variable=self.var1).pack(pady=(15, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Rotate by 180 degrees", width=27, variable=self.var9).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Rotate by 270 degrees", width=27, variable=self.var3).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Randomly flip left or right", width=27, variable=self.var4).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Randomly flip up or down", width=27, variable=self.var5).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Randomly change hue", width=27, variable=self.var6).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Randomly change saturation", width=27, variable=self.var7).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Randomly change brightness", width=27, variable=self.var8).pack(pady=(5, 0), padx=(60, 0))
        ttk.Checkbutton(tab2, text="Randomly change contrast", width=27, variable=self.var10).pack(pady=(5, 0), padx=(60, 0))

        self.button3 = ttk.Button(tab2, text="Deselect all", command=lambda: self.clearcheckboxes())
        self.button3.pack(pady=(20, 0), padx=(60, 0))
        self.button4 = ttk.Button(tab2, text="Delete augmentations in Training directory", command=lambda: self.delete_augmentations())
        self.button4.pack(pady=(15, 0), padx=(60, 0))


        # *********************** TAB 3 *****************
        # Creates a table for directorys in the training path and allow to select labes for each directory
        self.frame = Frame(tab3)
        self.frame.pack()


        # *********************** TAB 4 *****************
        # Allows users to configure the wanted preprocess options
        self.frame2 = Frame(tab4)
        self.frame2.pack()

        lb = Label(self.frame2, text="Desired Input Dimensions for Neural Network:       ")
        lb.grid(column=0, row=0)
        lb = Label(self.frame2, text="width")
        lb.grid(column=1, row=0)
        txt = Entry(self.frame2, textvariable=self.standardvalue_width, width=10)
        txt.grid(column=2, row=0)
        lb = Label(self.frame2, text="   height")
        lb.grid(column=3, row=0)
        txt = Entry(self.frame2, textvariable=self.standardvalue_height, width=10)
        txt.grid(column=4, row=0)

        lb = Label(self.frame2, text="")
        lb.grid(column=0, row=1, pady=(15, 0))
        lb = Label(self.frame2, text="Yes")
        lb.grid(column=1, row=1, pady=(15, 0))
        lb = Label(self.frame2, text="No")
        lb.grid(column=2, row=1, pady=(15, 0))

        lb = Label(self.frame2, text="Grayscale Images?")
        lb.grid(column=0, row=2)
        self.gss = []
        for i in range(2):
            self.gss.append(ttk.Checkbutton(self.frame2, onvalue=i, variable=var))
            self.gss[i].grid(column=i + 1, row=2)

        lb = Label(self.frame2, text="")
        lb.grid(column=0, row=3, pady=(15, 0))
        lb = Label(self.frame2, text="Yes")
        lb.grid(column=1, row=3, pady=(15, 0))
        lb = Label(self.frame2, text="No")
        lb.grid(column=2, row=3, pady=(15, 0))

        lb = Label(self.frame2, text="Normalize the pixel values between 0 and 1?")
        lb.grid(column=0, row=4)
        self.gs = []
        for i in range(2):
            self.gs.append(ttk.Checkbutton(self.frame2, onvalue=i, variable=self.var2))
            self.gs[i].grid(column=i + 1, row=4)

        # lb = Label(tab4, text="Desired Input Dimensions for Neural Network:       ")
        # lb.pack(side=tkinter.LEFT, pady=0, padx=60)
        #
        # lb2 = Label(tab4, text="width")
        # lb2.pack(side=tkinter.LEFT, pady=0, padx=15)
        # txt2 = Entry(tab4, textvariable=self.standardvalue_width, width=10)
        # txt2.pack(side=tkinter.LEFT, pady=0, padx=1)
        #
        # lb3 = Label(tab4, text="  height")
        # lb3.pack(side=tkinter.LEFT, pady=0, padx=15)
        # txt3 = Entry(tab4, textvariable=self.standardvalue_height, width=10)
        # txt3.pack(side=tkinter.LEFT, pady=0, padx=1)
        #
        # # lb = Label(self.frame2, text="")
        # # lb.pack(side=tkinter.TOP)
        # lb = Label(tab4, text="Yes")
        # lb.pack(side=tkinter.BOTTOM, pady=80, padx=1)
        # # lb = Label(tab4, text="No")
        # # lb.pack(pady=10)


        # *********************** TAB 5 *****************
        # Allows users to configure the wanted trainings options
        self.frame3 = Frame(tab5)
        self.frame3.pack()

        lb = Label(self.frame3, text="Choose validation size in %      ")

        lb.grid(column=0, row=0)

        txt = Spinbox(self.frame3, from_=1, to=99, width=5, textvariable=self.standardvalue_validationsize)

        txt.grid(column=1, row=0)

        lb = Label(self.frame3, text="Choose number of Epochs      ")

        lb.grid(column=0, row=1, pady=(15, 0))

        txt = Spinbox(self.frame3, from_=1, to=999, width=5, textvariable=self.standardvalue_epochs)

        txt.grid(column=1, row=1, pady=(15, 0))

        lb = Label(self.frame3, text="Choose batch size      ")

        lb.grid(column=0, row=2, pady=(15, 0))

        txt = Spinbox(self.frame3, from_=1, to=9999, width=5, textvariable=self.standardvalue_batchsize)

        txt.grid(column=1, row=2, pady=(15, 0))


        # *********************** TAB 6 *****************
        # Allows users to choose wether a predefined model structure is used or a custom model structure
        lb = Label(tab6, width=35, text="Options for previous model files: ")
        lb.pack(pady=(0, 10))

        main_menu2 = ttk.OptionMenu(tab6, clicked, "Don't delete any previous model files", "Don't delete any previous model files",
                                    "Delete oldest existing model file", "Delete all existing model files", command=self.callback_delete)
        main_menu2.pack(pady=(0, 70))
        ttk.Button(tab6, width=35, text="Customize model structure yourself", command=lambda: self.changevariable(0)).pack(pady=(0, 10))
        ttk.Button(tab6, width=35, text="Use predefined model structure", command=lambda: self.changevariable(1)).pack(pady=(0, 0))


        # *********************** TAB 7 *****************
        # Allows user to customize a model structure
        wrapper = LabelFrame(tab7)

        self.canvas = Canvas(wrapper)
        self.canvas.pack(side=LEFT, fill="both", expand="yes")
        yscrollbar = Scrollbar(wrapper, orient="vertical", command=self.canvas.yview)
        yscrollbar.pack(side=RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=yscrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.frame5 = Frame(self.canvas)
        self.frame5.bind("<Configure>", self.reset_scrollregion)
        self.frame5.bind_all("<MouseWheel>", self.on_mousewheel)  # Enable scrolling with mouse wheel
        self.canvas.create_window((0, 0), window=self.frame5, anchor="nw")
        wrapper.pack(fill="both", expand="yes")

        lb = Label(self.frame5, text="Choose number of neurons for first 'Conv2D' layer: ")
        lb.grid(column=0, row=0)
        self.txt1 = Entry(self.frame5, textvariable=standardvalue_conv2d, width=10)
        self.txt1.grid(column=1, row=0)
        lb = Label(self.frame5, text="Choose strides for first 'Conv2D' layer: ")
        lb.grid(column=0, row=1)
        self.txt2 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_one, width=10)
        self.txt2.grid(column=1, row=1)
        self.txt3 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_two, width=10)
        self.txt3.grid(column=2, row=1)
        lb = Label(self.frame5, text="Choose activation function for first 'Conv2D' layer: ")
        lb.grid(column=0, row=2)
        main_menu3 = ttk.OptionMenu(self.frame5, clicked2, "Relu", "Relu", "Sigmoid", "Softmax", "Softplus", "Softsign", "Tanh", "Selu", "Elu", "Exponential",
                                    command=self.callback_firstconv2d)
        main_menu3.grid(column=1, row=2)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=3, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose pooling size for first the 'MaxPooling2D' layer: ")
        lb.grid(column=0, row=4)
        self.txt4 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_one, state=DISABLED, width=10)
        self.txt4.grid(column=1, row=4)
        self.txt5 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_two, state=DISABLED, width=10)
        self.txt5.grid(column=2, row=4)
        self.button1 = ttk.Button(self.frame5, text="Use Layer", command=lambda: self.change_entry_state(self.button1, [self.txt4, self.txt5]))
        self.button1.grid(column=3, row=4)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=5, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose number of neurons for second 'Conv2D' layer: ")
        lb.grid(column=0, row=6)
        self.txt6 = Entry(self.frame5, textvariable=standardvalue_conv2d2, state=DISABLED, width=10)
        self.txt6.grid(column=1, row=6)
        lb = Label(self.frame5, text="Choose strides for second 'Conv2D' layer: ")
        lb.grid(column=0, row=7)
        self.txt7 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_one2, state=DISABLED, width=10)
        self.txt7.grid(column=1, row=7)
        self.txt8 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_two2, state=DISABLED, width=10)
        self.txt8.grid(column=2, row=7)
        lb = Label(self.frame5, text="Choose activation function for second 'Conv2D' layer: ")
        lb.grid(column=0, row=8)
        main_menu4 = ttk.OptionMenu(self.frame5, clicked3, "Relu", "Relu", "Sigmoid", "Softmax", "Softplus", "Softsign", "Tanh", "Selu", "Elu", "Exponential",
                                    command=self.callback_secondconv2d)
        main_menu4.grid(column=1, row=8)
        self.button2 = ttk.Button(self.frame5, text="Use Layer", command=lambda: self.change_entry_state(self.button2, [self.txt6, self.txt7, self.txt8, self.button3, self.button4]))
        self.button2.grid(column=3, row=7)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=9, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose pooling size for the second 'MaxPooling2D' layer: ")
        lb.grid(column=0, row=10)
        self.txt9 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_one2, state=DISABLED, width=10)
        self.txt9.grid(column=1, row=10)
        self.txt10 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_two2, state=DISABLED, width=10)
        self.txt10.grid(column=2, row=10)
        self.button3 = ttk.Button(self.frame5, text="Use Layer", state=DISABLED, command=lambda: self.change_entry_state(self.button3, [self.txt9, self.txt10]))
        self.button3.grid(column=3, row=10)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=11, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose number of neurons for third 'Conv2D' layer: ")
        lb.grid(column=0, row=12)
        self.txt11 = Entry(self.frame5, textvariable=standardvalue_conv2d3, state=DISABLED, width=10)
        self.txt11.grid(column=1, row=12)
        lb = Label(self.frame5, text="Choose strides for third 'Conv2D' layer: ")
        lb.grid(column=0, row=13)
        self.txt12 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_one3, state=DISABLED, width=10)
        self.txt12.grid(column=1, row=13)
        self.txt13 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_two3, state=DISABLED, width=10)
        self.txt13.grid(column=2, row=13)
        lb = Label(self.frame5, text="Choose activation function for third 'Conv2D' layer: ")
        lb.grid(column=0, row=14)
        main_menu5 = ttk.OptionMenu(self.frame5, clicked4, "Relu", "Relu", "Sigmoid", "Softmax", "Softplus", "Softsign", "Tanh", "Selu", "Elu", "Exponential",
                                    command=self.callback_thirdconv2d)
        main_menu5.grid(column=1, row=14)
        self.button4 = ttk.Button(self.frame5, text="Use Layer", state=DISABLED, command=lambda: self.change_entry_state(self.button4, [self.txt11, self.txt12, self.txt13, self.button5, self.button6]))
        self.button4.grid(column=3, row=13)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=15, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose pooling size for the third 'MaxPooling2D' layer: ")
        lb.grid(column=0, row=16)
        self.txt14 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_one3, state=DISABLED, width=10)
        self.txt14.grid(column=1, row=16)
        self.txt15 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_two3, state=DISABLED, width=10)
        self.txt15.grid(column=2, row=16)
        self.button5 = ttk.Button(self.frame5, text="Use Layer", state=DISABLED, command=lambda: self.change_entry_state(self.button5, [self.txt14, self.txt15]))
        self.button5.grid(column=3, row=16)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=18, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose number of neurons for fourth 'Conv2D' layer: ")
        lb.grid(column=0, row=19)
        self.txt16 = Entry(self.frame5, textvariable=standardvalue_conv2d4, state=DISABLED, width=10)
        self.txt16.grid(column=1, row=19)
        lb = Label(self.frame5, text="Choose strides for fourth 'Conv2D' layer: ")
        lb.grid(column=0, row=20)
        self.txt17 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_one4, state=DISABLED, width=10)
        self.txt17.grid(column=1, row=20)
        self.txt18 = Entry(self.frame5, textvariable=standardvalue_conv2d_stride_two4, state=DISABLED, width=10)
        self.txt18.grid(column=2, row=20)
        lb = Label(self.frame5, text="Choose activation function for fourth 'Conv2D' layer: ")
        lb.grid(column=0, row=21)
        main_menu6 = ttk.OptionMenu(self.frame5, clicked5, "Relu", "Relu", "Sigmoid", "Softmax", "Softplus", "Softsign", "Tanh", "Selu", "Elu", "Exponential",
                                    command=self.callback_fourthconv2d)
        main_menu6.grid(column=1, row=21)
        self.button6 = ttk.Button(self.frame5, text="Use Layer", state=DISABLED, command=lambda: self.change_entry_state(self.button6, [self.txt16, self.txt17, self.txt18, self.button7]))
        self.button6.grid(column=3, row=20)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=22, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose pooling size for the fourth 'MaxPooling2D' layer: ")
        lb.grid(column=0, row=23)
        self.txt19 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_one4, state=DISABLED, width=10)
        self.txt19.grid(column=1, row=23)
        self.txt20 = Entry(self.frame5, textvariable=standardvalue_maxpooling_stride_two4, state=DISABLED, width=10)
        self.txt20.grid(column=2, row=23)
        self.button7 = ttk.Button(self.frame5, text="Use Layer", state=DISABLED, command=lambda: self.change_entry_state(self.button7, [self.txt19, self.txt20]))
        self.button7.grid(column=3, row=23)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=24, sticky="ew", pady=(15, 15))

        lb = Label(self.frame5, text="Choose dropout ratio in %: ")
        lb.grid(column=0, row=25, pady=(15, 0))
        self.txt21 = Entry(self.frame5, textvariable=self.standardvalue_dropoutratio1, state=DISABLED, width=10)
        self.txt21.grid(column=1, row=25, pady=(15, 0))
        lb = Label(self.frame5, text="")
        lb.grid(column=2, row=25, pady=(15, 0))
        button8 = ttk.Button(self.frame5, text="Use Layer", command=lambda: self.change_entry_state(button8, [self.txt21]))
        button8.grid(column=3, row=25)

        separator = ttk.Separator(self.frame5, orient='horizontal')
        separator.grid(column=0, row=26, sticky="ew", pady=(15, 15))
        lb23 = Label(self.frame5, text="Choose the amount of hidden layers: ")
        lb23.grid(column=0, row=27)

        self.txt22 = Entry(self.frame5, textvariable=standardvalue_hiddenlayers, state=DISABLED, width=10)
        self.txt22.grid(column=1, row=27)
        self.button13 = ttk.Button(self.frame5, text="Add Hidden Layers", state=DISABLED, command=lambda: [lb23.destroy(), button9.destroy(),
                                                                                                                 self.txt22.destroy(),
                                                                                                                 self.button13.destroy(),
                                                                                                                 self.customize_hiddenlayers(int(standardvalue_hiddenlayers.get()), 27)])
        self.button13.grid(column=2, row=27)
        button9 = ttk.Button(self.frame5, text="Use Layer", command=lambda: self.change_entry_state(button9, [self.txt22, self.button13]))
        button9.grid(column=3, row=27)

        # *********************** TAB 8 *****************
        # Allows user to train a model with the selected configurations
        self.firstconv2d = "relu"
        self.secondconv2d = "relu"
        self.thirdconv2d = "relu"
        self.fourthconv2d = "relu"
        self.settings["model_save"] = "1"

        lb = Label(tab8, text="Choose execution mode: ")
        lb.pack(pady=(0, 10))

        main_menu7 = ttk.OptionMenu(tab8, clicked6, "Automatic", "Automatic",
                                    "Use GPU for training and CPU for predicting with memory growth enabled for the GPU",
                                    "Use GPU for training and predicting", "Force CPU for training and predicting",
                                    command=self.callback_excution_options)
        main_menu7.pack(pady=(0, 140))
        self.button30 = ttk.Button(tab8, text="Start Training", command=lambda: self.updatesettings())
        self.button30.pack()

        # *************************************t**********
        self.tab_control.pack(expand=1, fill='both')
        self.window.mainloop()


