
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

from naturalScenesData import NaturalScenes
from Model import Model

def plot_performance(training):
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
        plt.savefig("plots/model_"+str(training["model_id"]))


module_list = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, "same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 6),
        nn.Dropout(),
        nn.LogSoftmax(dim=1)
    ),
])


data = NaturalScenes('D:/projects/data/NaturalScenes')

model = Model(1, module_list)
model.optimizer = optim.Adam(model.parameters(), lr=0.1)
model.loss_fn = nn.NLLLoss()


training_output = model.learn_test(data.train_loader, data.valid_loader, data.test_loader, num_epochs=50)
plot_performance(training_output)

