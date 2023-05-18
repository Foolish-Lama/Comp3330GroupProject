from torch import nn, optim
import os
import matplotlib.pyplot as plt
import numpy as np

from jsonFile import json_file
from Model import Model

class Folder():
    #api used to optimize ANNs for data
    # saves everything to output folder

    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder) 

        self.model_f = json_file(self.output_folder+"/models.json")
        self.trainings_f = json_file(self.output_folder+"/trainings.json")
        self.plots_f = self.output_folder+"/plots"
        if not os.path.exists(self.plots_f):
            os.mkdir(self.plots_f) 
            
    # models
    def post_model(self, description, module_list, optimizer_class, optimizer_kwargs, loss_fn_class):
        new_model = {
            "description": description,
            "module_list": module_list,
            "optimizer_class": optimizer_class,
            "optimizer_kwargs ": optimizer_kwargs,
            "loss_fn_class": loss_fn_class,
        }
        return self.model_f.add(new_model)
    
    def post_model(self, new_model):
        return self.model_f.add(new_model)

    def get_model(self, id):
        return self.model_f.get(id)
     
    def update_model(self, id, updates):
        return self.model_f.eupdate(id, updates)

    # training data
    def post_training(self, model_id, data_description, losses, accuracys, train_duration, test_duration, test_loss, test_accuracy):
        new_training = {
                "model_id": model_id,
                "data_description": data_description,
                "losses" : losses,
                "accuracys": accuracys,
                "train_duration": train_duration,
                "test_duration": test_duration,
                "test_loss" : test_loss,
                "test_accuracy": test_accuracy
            }
        return self.trainings_f.add(new_training)
    
    def post_training(self, new_training):
        return self.trainings_f.add(new_training)

    def get_training(self, id):
        return self.trainings_f.get(id)
    
    def get_training_by_model(self, model_id):
        return self.trainings_f.find("model_id", model_id)


    def plot_performance(self, id):
        training = self.get_training(id)
        plt.clf()
        x = np.array([c for c, _ in enumerate(training["losses"], start=1)])
        y = np.array([v for v in training["losses"]])

        plt.subplot(2, 1, 1)
        plt.plot(x, y)
        plt.ylabel("loss")

        x = np.array([c for c, _ in enumerate(training["accuracys"], start=1)])
        y = np.array([v for v in training["accuracys"]])

        plt.subplot(2, 1, 2)
        plt.plot(x, y)
        plt.ylabel("accuracy")
        plt.ylim(0, 1)

        plt.suptitle("model "+str(training["model_id"])+" performance "+str(training["id"]))
        plt.savefig(self.plots_f+"/model_"+str(training["model_id"])+"_performance_"+str(training["id"]))


    optimizers = {
            "Adam": optim.Adam
        }

    loss_fns = {
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "NLLLoss": nn.NLLLoss
        }

    layers = {
            "Conv2d": nn.Conv2d,
            "BatchNorm2d": nn.BatchNorm2d,
            "ReLU": nn.ReLU,
            "MaxPool2d": nn.MaxPool2d,
            "Flatten": nn.Flatten,
            "Linear": nn.Linear,
            "Dropout": nn.Dropout,
            "Sequential": nn.Sequential,
            "LogSoftmax ": nn.LogSoftmax 

        }

    layer_ks = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "num_features",
        "in_features",
        "out_features"
    ]

    optimizer_ks = [
        "lr",
        "weight_decay",
        "momentum",
        "dampening"
    ]

    def translate_layer_dic(self, layer_dic):
        layer_class = self.layers[layer_dic["eclass"]]
        layer_kwargs = layer_dic["kwargs"]
        layer = layer_class(**layer_kwargs)
        return layer
    
    def translate_layer(self, layer):
        layer_class = None
        for k, v in self.layers.items():
            if v == type(layer):
                layer_class = k
        layer_kwargs = vars(layer)
        layer_kwargs = {k: v for k, v in layer_kwargs.items() if k in self.layer_ks}

        layer_dic = {
            "eclass": layer_class,
            "kwargs": layer_kwargs
        }
        return layer_dic

    def load_model(self, model_id):
        model_template = self.get_model(model_id)
        
        model_id = model_template["id"]

        module_list = nn.ModuleList()
        for layer_dic in model_template["module_list"]:
            if layer_dic["eclass"] == "Sequential":
                sub_layers = []
                for sub_layer_dic in layer_dic["args"]:
                    sub_layers.append(self.translate_layer_dic(sub_layer_dic))
                module_list.append(nn.Sequential(*sub_layers))
            else:
                module_list.append(self.translate_layer_dic(layer_dic))


        optimizer_class = self.optimizers[model_template["optimizer_class"]]
        optimizer_kwargs = model_template["optimizer_kwargs"]
        loss_fn_class = self.loss_fns[model_template["loss_fn_class"]]


        model = Model(model_id, module_list)
        model.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        model.loss_fn = loss_fn_class()

        return model
    
    def upload_model(self, model, description=None):
        module_list = []
        for layer in model.module_list:
            if type(layer) == nn.Sequential:
                layer_args = []
                for sub_layer in layer:
                    layer_args.append(self.translate_layer(sub_layer))

                layer_dic = {
                    "eclass": "Sequential",
                    "args": layer_args
                }
                module_list.append(layer_dic)
            else:
                module_list.append(self.translate_layer(layer))

        optimizer_class = [k for k, v in self.optimizers.items() if type(model.optimizer) == v ][0]
        optimizer_kwargs  = {k:v for k, v in vars(model.optimizer)["defaults"].items() if k in self.optimizer_ks}
        loss_fn_class = [k for k, v in self.loss_fns.items() if v == model.loss_fn][0]

        model_template = {
            "description": description,
            "module_list": module_list,
            "optimizer_class": optimizer_class,
            "optimizer_kwargs": optimizer_kwargs,
            "loss_fn_class": loss_fn_class,
        }
        return self.post_model(model_template)



class App(Folder):

    current_model = None

    train_loader  = None
    valid_loader = None
    test_loader = None
    loaders = [train_loader, valid_loader, test_loader]

    def __init__ (self, output_folder):
        super().__init__(output_folder)

    def set_data(self, train, valid, test):
        self.train_loader = train
        self.valid_loader = valid
        self.test_loader = test
    
    def set_model(self, id):
        self.current_model = self.load_model(id)
    
    def test_model(self):
        self.current_model.test_model(self.train_loader, self.valid_loader, self.test_loader)
    
    def learn_test_model(self, num_epocs=10):
        training = self.current_model.learn_test(self.train_loader, self.valid_loader, self.test_loader, num_epocs)
        trainig = self.post_training(training)
        self.plot_performance(trainig["id"])

    def test_all_models(self):
        for m in self.model_f.content:
            self.set_model(m["id"])
            self.test_model()
    
    def learn_test_all_models(self, num_epochs=10):
         for m in self.model_f.content:
            self.set_model(m["id"])
            self.learn_test_model(num_epochs)
