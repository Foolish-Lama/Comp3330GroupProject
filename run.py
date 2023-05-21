
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import ModuleLists



using2d = ModuleLists.Using2d()


data = NaturalScenesSubset('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')

for ml, title, i in zip(using2d.all_module_lists, using2d.all_module_titles, range(1, len(using2d.all_module_lists))):
    model = Model(i, 'using2d', ml, using2d.optimizer_class, using2d.loss_fn_class)
    model.run(*data.loaders, title=title)

data = NaturalScenes('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')

loss_fn_class = nn.NLLLoss
optimizer_class = optim.Adam

module_list = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.Tanh(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.BatchNorm2d(32),
        nn.Tanh(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, "same"),
        nn.BatchNorm2d(64),
        nn.Tanh(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 6),
        nn.Dropout(),
        nn.LogSoftmax(dim=1)
    ),
])

model = Model(i, 'fullData', module_list, optimizer_class, loss_fn_class)
model.run(*data.loaders, title="theOne", num_epochs=50)
