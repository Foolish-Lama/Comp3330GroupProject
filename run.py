
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes
from Model import Model


module_list_1 = nn.ModuleList([
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

module_list_2 = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(4)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 1296),
        nn.Linear(1296, 36),
        nn.Linear(36, 6),
        nn.ReLU()
    )
])



data = NaturalScenes('D:/projects/data/NaturalScenes/seg_train')

model = Model(1, module_list_1)
model.optimizer = optim.Adam(model.parameters(), lr=0.1)
model.loss_fn = nn.CrossEntropyLoss()


model.run(*data.loaders, num_epochs=15)


