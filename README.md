# Comp3330GroupProject

created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

Files:
    run.py : is used as a working file to do tests
    Model.py : is the a model class that inherits from nn.Module
    naturalScenesData.py : prepares the data
    tests.py : are the classes used for testing
    bestModel.py :  is the best model
    jsonFile.py : just used to create json files

Usage:
    1. initialize a NaturalScenes data class, giving it the approitat path to the folder

    2. initialize model with paramaters:
        id: an int to keep track of models
        outputFolder: the folder the results will go into
        module_list: a nn.ModuleList()
        optimizer_class: a optim class
            - you can also reset this after inilization to specify a learning rate
        loss_fn_class: nn loss function class
    
    3. calling methods:
        .test_model() : just makes sure all the other methods run
        .run(): trains the model and then tests it, saving the state_dict, the results in a json file, and plots a plot
            - it takes the data loaders, a title for the plot and number of epochs as arguments