
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import ModuleLists



optimizers = ModuleLists.Optimizers()


data = NaturalScenesSubset('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')

module_list = []

for op, title, i in zip(optimizers.all_optimzers, optimizers.all_optimzers_titles, range(1, len(optimizers.all_optimzers))):
    model = Model(i, 'optimizers', optimizers.module_list, op, optimizers.loss_fn_class)
    model.run(*data.loaders, title=title)

data = NaturalScenes('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')
