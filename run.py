
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import ModuleLists

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

title = "bestModel2"
id = 2

model = Model(id, outputFile, module_list, optimizer_class, loss_fn_class)
model.run(*data.loaders, title=title, num_epochs=100)


failed_tests = []

ActivationFunctions = ModuleLists.ActivationFunctions()

for ml, title, i in zip(ActivationFunctions.all_module_lists, ActivationFunctions.all_module_titles, range(1, len(ActivationFunctions.all_module_lists))):
    try:
        model = Model(i, 'ActivationFunctionsFullData', ml, ActivationFunctions.optimizer_class, ActivationFunctions.loss_fn_class)
        model.run(*data.loaders, title=title, num_epochs=10)
    except:
        s = str(i) + " " + title
        failed_tests.append(s)
        print(s + "failed")


Optimizers = ModuleLists.Optimizers()

for op, title, i in zip(Optimizers.all_optimzers, Optimizers.all_optimzers_titles, range(1, len(Optimizers.all_optimzers))):
    try:
        model = Model(i, 'OptimizersFullData', Optimizers.module_list, op, Optimizers.loss_fn_class)
        model.run(*data.loaders, title=title, num_epochs=5)
    except:
        s = str(i) + " " + title
        failed_tests.append(s)
        print(s + "failed")
