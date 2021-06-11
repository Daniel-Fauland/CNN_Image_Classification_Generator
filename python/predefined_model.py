class Predefined_model():
    def initialize(self, settings):
        # --- These are the settings for the predefined model ---
        # This preset uses 2 pairs of conv2d and max poolings layers, a dropout layer, 1 hidden layer and another dropout layer
        # Activation functions: 1 = relu, 2 = sigmoid, 3 = softmax, 4 = softplus, 5 = softsign, 6 = tanh, 7 = selu, 8 = elu, 9 = exponential

        settings["count_layers"] = 2  # Choose number of convolutional layers (min number: 1)
        settings["num_neurons_1"] = "64"  # Choose number of neurons for first convolutional layer
        settings["num_neurons_2"] = "64"  # Choose number of neurons for second convolutional layer
        settings["activation_type_1"] = "1"  # Choose activation function for first convolutional layer
        settings["activation_type_2"] = "1"  # Choose activation function for second convolutional layer
        settings["strides_neurons_1"] = "3 3"  # Choose strides for first convolutional layer
        settings["strides_neurons_2"] = "3 3"  # Choose strides for second convolutional layer

        # This array has to be the same length as number of conv. layer. Specify with 'y' or 'n' the amount and position of the MaxPooling layers
        settings["pooling_layers"] = ["y", "y"]
        settings["max_pool_1"] = "2 2"  # Choose pooling size for the first MaxPooling layer
        settings["max_pool_2"] = "2 2"  # Choose pooling size for the second MaxPooling layer
        settings["dropout_1"] = "20"  # Choose dropout ratio for first dropout layer. If you don't want a dropout layer set the value to "0"
        settings["num_hidden_layers"] = 1  # Choose number of hidden/dense layers (min number: 0)
        settings["hidden_layer_1"] = "64"  # Choose number of neurons for the first hidden layer
        settings["hidden_layer_activation_1"] = "1"  # Choose activation function for the first hidden layer
        settings["dropout_2"] = "20"  # Choose dropout ratio for second dropout layer. If you don't want a dropout layer set the value to "0"
        return settings

    # --- Example for a different model architecture ---
    # 1. Conv2d layer with 32 neurons and (3 3) Strides and relu activation function
    # 2. Second Conv2d layer with 64 neurons and (3 3) Strides and relu activation function
    # 3. Max Pooling layer with a pool size of (4 4) and relu activation function
    # 4. Hidden layer with 64 neurons and relu activation function
    # 5. Hidden layer with 32 neurons and exponential activation function
    # 6. Dropout layer with a dropout rate of 30 %

    # settings["count_layers"] = 2  # Choose number of convolutional layers (min number: 1)
    # settings["num_neurons_1"] = "32"  # Choose number of neurons for first convolutional layer
    # settings["num_neurons_2"] = "64"  # Choose number of neurons for second convolutional layer
    # settings["activation_type_1"] = "1"  # Choose activation function for first convolutional layer
    # settings["activation_type_2"] = "1"  # Choose activation function for second convolutional layer
    # settings["strides_neurons_1"] = "3 3"  # Choose strides for first convolutional layer
    # settings["strides_neurons_2"] = "3 3"  # Choose strides for second convolutional layer
    #
    # # This array has to be the same length as number of conv. layer. Specify with 'y' or 'n' the amount and position of the MaxPooling layers
    # settings["pooling_layers"] = ["n", "y"]
    # settings["max_pool_1"] = "4 4"  # Choose pooling size for the first MaxPooling layer
    # settings["num_hidden_layers"] = 2  # Choose number of hidden/dense layers (min number: 0)
    # settings["hidden_layer_1"] = "64"  # Choose number of neurons for the first hidden layer
    # settings["hidden_layer_activation_1"] = "1"  # Choose activation function for the first hidden layer
    # settings["hidden_layer_2"] = "32"  # Choose number of neurons for the first hidden layer
    # settings["hidden_layer_activation_2"] = "9"  # Choose activation function for the first hidden layer
    # settings["dropout_2"] = "30"  # Choose dropout ratio for second dropout layer. If you don't want a dropout layer set the value to "0"
