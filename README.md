# n_layer_simple_neural_network
N layer simple neural networks implementation with mini-batch and Adam optimization algorithm.
The activation function of the last layer is sigmoid.
The program saves parameters which are weights and bias as .csv file like W1.csv, W2.csv, b1.csv, b2.csv

There are a couple of things must be done before using the code

First On the project, the following libraries are used. Must be download these libraries before using the code
pandas
numpy
sklearn
matplotlib
scipy


Second The following .csv file names must be created.

        x_train.csv                train dataset
        y_train.csv   

        x_cross_validation.csv     Cross validation(development) dataset    
        y_cross_validation.csv    

        x_test.csv                 test dataset 
        y_test.csv

or go to the loading.py and change the file paths

Last thing, "layers" information must be givin the main() method.

        # this example code shows how to initialize 4 layer neural networks
        # last layer is added automatically with sigmoid function 
        # by using y_train shape.
        # layers = [layer_1, layer_2, layer_3, ..., layer_x]
        # layer_x = tuple contains layer information
        # layer_x = (number_of_units, activation_function_name_string)
        # activation_function_name_string = "relu", "tanh" or "sigmoid"
        import main
        layers = [(25, "tanh"), (10, "relu"), (5, "tanh")]   
        result = main.main(layers)


