
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes
from Model import Model



data = NaturalScenes('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')


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

outputFile = "bestModel"
optimizer_class = optim.NAdam
loss_fn_class = nn.NLLLoss

title = "bestModel1"
i = 1

model = Model(i, outputFile, module_list, optimizer_class, loss_fn_class)
model.run(*data.loaders, title=title, num_epochs=5)

