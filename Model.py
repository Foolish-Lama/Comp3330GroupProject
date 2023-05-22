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

    def __init__(self, id, output_path, module_list, optimizer_class, loss_fn_class):
        super(Model, self).__init__()
        self.id = id

        self.output_folder = output_path
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder) 
        self.plots_folder = self.output_folder+"/plots"
        if not os.path.exists(self.plots_folder):
            os.mkdir(self.plots_folder)
        self.state_dic_folder = self.output_folder+"/state_dics"
        if not os.path.exists(self.state_dic_folder):
            os.mkdir(self.state_dic_folder) 
        self.performances_file = json_file(self.output_folder+"/performances.json")

        self.module_list = module_list
        self.optimizer = optimizer_class(self.parameters(), lr=1)
        self.loss_fn = loss_fn_class()
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
        train_losses = []
        train_accs = []
        eval_losses = []
        eval_accs = []
        best_accuracy = 0
        best_state_dic = None
        for epoch in range(num_epochs):
            start_t = time()
            train_loss, train_acc = self.evolve(train_loader)
            eval_loss, eval_acc = self.validate(eval_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            if eval_acc > best_accuracy:
                best_accuracy = eval_acc
                best_state_dic = self.state_dict()
                path = "{}/abc_model_{}_{}_uuid_{}".format(self.state_dic_folder, eval_acc, self.id, uuid.uuid1().hex)
                torch.save(self.state_dict(), path)
        
        print()
        print("Model: {}".format(self.id))
        print("Epochs: {}".format(num_epochs))
        print("Best Accuracy: {:.2f}%".format(best_accuracy*100))
        print("Total training time: {:.2f}".format(time()-start_t))
        return {
            "train_losses": train_losses,
            "train_accuracys": train_accs,
            "eval_losses": eval_losses,
            "eval_accuracys": eval_accs,
            "duration": time()-start_t,
            "best_accuracy": best_accuracy,
            "best_state_dic": best_state_dic
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
        path = "{}/model_{}_{}_uuid_{}".format(self.state_dic_folder, learn["best_accuracy"], self.id, uuid.uuid1().hex)
        torch.save(learn["best_state_dic"], path)
        self.load_state_dict(torch.load(path))
        test = self.test(test_loader)

        print("total time: ", str(time()-start_t))

        performance = {
            "train_losses" : learn["train_losses"],
            "train_accuracys": learn["train_accuracys"],
            "eval_losses" : learn["eval_losses"],
            "eval_accuracys": learn["eval_accuracys"],
            "train_duration": learn["duration"],
            "best_accuracy": learn["best_accuracy"],
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
        x = np.array([c for c, _ in enumerate(performance["train_losses"], start=1)])
        y1 = np.array([v for v in performance["train_losses"]])
        y2 = np.array([v for v in performance["eval_losses"]])

        plt.subplot(2, 1, 1)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.legend(["train", "val"])

        plt.ylabel("loss")

        x = np.array([c for c, _ in enumerate(performance["train_accuracys"], start=1)])
        y1 = np.array([v for v in performance["train_accuracys"]])
        y2 = np.array([v for v in performance["eval_accuracys"]])

        plt.subplot(2, 1, 2)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.legend(["train", "val"])
        plt.ylabel("accuracy")
        plt.ylim(0, 1)

        plt.suptitle("model {}: {}".format(self.id, title))

        plt.savefig("{}/model_{}_uuid_{}".format(self.plots_folder, self.id, uuid.uuid1().hex))
