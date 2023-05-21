# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com


import torch
from torch import nn
from time import time
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os

from jsonFile import json_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

class Model(nn.Module):

    def __init__(self, id, output_path, module_list, optimizer, loss_fn):
        super(Model, self).__init__()
        self.id = id

        self.output_folder = output_path
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder) 
        self.plots_folder = self.output_folder+"/plots"
        if not os.path.exists(self.plots_folder):
            os.mkdir(self.plots_folder) 
        self.performances_file = json_file(self.output_folder+"/performances.json")

        self.module_list = module_list
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.to(device)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)

        return x
    
    def evolve(self, data_loader):
        e_loss, e_accuracy = 0, 0

        self.train()
        for d, l in data_loader:
            self.optimizer.zero_grad()

            d.to(device)
            l = l.to(device)

            output = self(d)

            loss = self.loss_fn(output, l)


            loss.backward()
            self.optimizer.step()

            e_accuracy += ((output.argmax(dim=1) == l).float().mean())/len(data_loader)
            e_loss += loss/len(data_loader)
        return e_loss.item(), e_accuracy.item()

    def validate(self, data_loader):
        e_accuracy, e_loss = 0, 0
        
        self.eval()
        with torch.no_grad():
            for d, l in data_loader:
                d = d.to(device)
                l = l.to(device)

                output = self(d)
                loss = self.loss_fn(output, l)
                


                e_accuracy += ((output.argmax(dim=1) == l).float().mean())/len(data_loader)
                e_loss += loss/len(data_loader)
        return e_loss.item(), e_accuracy.item()

    def learn(self, train_loader, eval_loader, num_epochs=10):
        start_t = time()
        eval_losses = []
        eval_accs = []
        for epoch in range(num_epochs):
            start_t = time()
            self.evolve(train_loader)
            eval_loss, eval_acc = self.validate(eval_loader)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
        
        print()
        print("Model: " + str(self.id))
        print("Epochs: " + str(num_epochs))
        print("Final Accuracy: " + str(eval_accs[len(eval_accs)-1]))
        print("Final Loss: " + str(eval_losses[len(eval_losses)-1]))
        print("Total training time: ", str(time()-start_t))
        return {
            "losses": eval_losses,
            "accuracys": eval_accs,
            "duration": time()-start_t
        }
    
    def test(self, test_loader):
        start_t = time()
        eval_loss, eval_acc = self.validate(test_loader)
        print('Test acc: {:.2f}%, loss: {:.4f}'.format(eval_acc*100, eval_loss))
        return {
            "loss": eval_loss,
            "accuracy": eval_acc,
            "duration": time()-start_t
        }
    
    def run(self, train_loader, valid_loader, test_loader, title='', num_epochs=10):
        start_t = time()

        learn = self.learn(train_loader, valid_loader, num_epochs)
        test = self.test(test_loader)

        print("total time: ", str(time()-start_t))

        performance = {
            "losses" : learn["losses"],
            "accuracys": learn["accuracys"],
            "train_duration": learn["duration"],
            "test_duration": test["duration"],
            "test_loss" : test["loss"],
            "test_accuracy": test["accuracy"]
            }
        self.performances_file.add(performance)
        self.plot_performance(performance, title)
        return performance

    def test_model(self, train_loader, valid_loader, test_loader):
        # will fuck with training, alittle
        print()
        print("testing model: "+str(self.id))
        self.learn(train_loader, valid_loader, 1)
        self.test(test_loader)
        print('successful')
    
    def plot_performance(self, performance, title=''):
        plt.clf()
        x = np.array([c for c, _ in enumerate(performance["losses"], start=1)])
        y = np.array([v for v in performance["losses"]])

        plt.subplot(2, 1, 1)
        plt.plot(x, y)
        plt.ylabel("loss")

        x = np.array([c for c, _ in enumerate(performance["accuracys"], start=1)])
        y = np.array([v for v in performance["accuracys"]])

        plt.subplot(2, 1, 2)
        plt.plot(x, y)
        plt.ylabel("accuracy")
        plt.ylim(0, 1)

        plt.suptitle("model {}: {}".format(self.id, title))

        plt.savefig("{}/model_{}_uuid_{}".format(self.plots_folder, self.id, uuid.uuid1().hex))
