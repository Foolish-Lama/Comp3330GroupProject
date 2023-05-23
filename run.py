
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import tests

data = NaturalScenes('E:/programming/data/NaturalScenes/seg_train', 'E:/programming/data/NaturalScenes/seg_test')


# model parameters
module_list = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.BatchNorm2d(32),
        nn.Sigmoid(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, "same"),
        nn.BatchNorm2d(64),
        nn.Sigmoid(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 6),
        nn.Dropout(),
        nn.LogSoftmax(dim=1)
    )
])

outputFolder = "learningRateFullData"
optimizer_class = optim.Adam
loss_fn_class = nn.NLLLoss


#testing learning rates
title = "lr 0.1"
id = 1
model = Model(id, outputFolder, module_list, optimizer_class, loss_fn_class)
model.optimizer = optimizer_class(model.parameters(), lr=0.1)
model.run(*data.loaders, title=title, num_epochs=10)

title = "lr 0.01"
id = 2
model = Model(id, outputFolder, module_list, optimizer_class, loss_fn_class)
model.optimizer = optimizer_class(model.parameters(), lr=0.01)
model.run(*data.loaders, title=title, num_epochs=10)

title = "lr 0.001"
id = 3
model = Model(id, outputFolder, module_list, optimizer_class, loss_fn_class)
model.optimizer = optimizer_class(model.parameters(), lr=0.001)
model.run(*data.loaders, title=title, num_epochs=10)

title = "lr 0.0001"
id = 4
model = Model(id, outputFolder, module_list, optimizer_class, loss_fn_class)
model.optimizer = optimizer_class(model.parameters(), lr=0.0001)
model.run(*data.loaders, title=title, num_epochs=10)


# testing best model
outputFolder = "BestModel"
optimizer_class = optim.Adam
loss_fn_class = nn.NLLLoss

id = 1
model = Model(id, outputFolder, module_list, optimizer_class, loss_fn_class)
model.optimizer = optimizer_class(model.parameters(), lr=0.0005)
for n in range(10):
    title = str(n*10) + " epochs"
    model.run(*data.loaders, title=title, num_epochs=10)
