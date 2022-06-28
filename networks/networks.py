import torch.nn as nn
import torch.nn.functional as F
import logging

from .utils import ReparamModule

class LeNet(ReparamModule):

    def __init__(self, state):
        logging.info(f'Build one LeNet network with [{state.init}({state.init_param})] init')
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, state.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out