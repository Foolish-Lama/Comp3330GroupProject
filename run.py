
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import ModuleLists



activationFunctions = ModuleLists.ActivationFunctions()



data = NaturalScenesSubset('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')

for module_list, title, i in zip(activationFunctions.all_module_lists, activationFunctions.all_module_titles, range(1, len(activationFunctions.all_module_lists))):
    model = Model(i, '/activationFunctions', module_list, optim.Adam, nn.NLLLoss)
    model.test_model(*data.loaders)
    model.run(*data.loaders, num_epochs=10)


