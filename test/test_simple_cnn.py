import unittest

from torch import nn
import torch.nn.functional as F
import pytorch2timeloop

from .test_configs import TMP_TEST_DIR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5, padding = 2)
        self.fc1 = nn.Linear(8 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TestSimpleCNN(unittest.TestCase):
    def setUp(self):
        self.net = Net()
        self.input_size = (1, 28, 28)
        self.batch_size = 1

    def test_simple_cnn_without_fc(self):
        pytorch2timeloop.convert_model(
            model=self.net,
            input_size=self.input_size,
            batch_size=self.batch_size,
            convert_fc=False,
            model_name='simple_cnn_without_fc',
            save_dir=TMP_TEST_DIR,
            exception_module_names=[]
        )

    def test_simple_cnn_with_fc(self):
        pytorch2timeloop.convert_model(
            model=self.net,
            input_size=self.input_size,
            batch_size=self.batch_size,
            convert_fc=True,
            model_name='simple_cnn_with_fc',
            save_dir=TMP_TEST_DIR,
            exception_module_names=[]
        )

class GroupedCNN(nn.Module):
    def __init__(self):
        super(GroupedCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, 5, groups=2, padding=2)

    def forward(self, x):
        return self.conv1(x)

class TestGroupedConv(unittest.TestCase):
    def setUp(self):
        self.net = GroupedCNN()
        self.input_size = (4, 28, 28)
        self.batch_size = 1

    def test_grouped_conv(self):
        pytorch2timeloop.convert_model(
            model=self.net,
            input_size=self.input_size,
            batch_size=self.batch_size,
            convert_fc=False,
            model_name='grouped_conv',
            save_dir=TMP_TEST_DIR,
            exception_module_names=[]
        )