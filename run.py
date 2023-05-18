from Model import Model
from naturalScenesData import NaturalScenes

import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim

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

        plt.suptitle("model "+str(training["model_id"]))
        plt.savefig(self.plots_f+"/model_"+str(training["model_id"]))

module_list_1 = nn.ModelList([
        

])

data = NaturalScenes()
model = Model(1, module_list_1)

model.test_model(data.train_loader, data.valid_loader, data.test_loader)
 