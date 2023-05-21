from torch import nn

class ActivationFunctions:

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
        ),
    ])

    module_list_3 = nn.ModuleList([
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

    module_list_4 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.Dropout(),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_5 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.Dropout(),
            nn.LogSoftmax(dim=1)
        ),
    ])

    all_module_lists = [
        module_list_1,
        module_list_2,
        module_list_3,
        module_list_3,
        module_list_4,
        module_list_5,
    ]

    all_module_titles = [
        "RELU",
        "Sigmoid",
        "Tanh",
        "LeakyReLU",
        "ELU",
    ]