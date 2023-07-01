import unittest

import torch
from torchvision.models import resnet18
import pytorch2timeloop

from .test_configs import TMP_TEST_DIR

class TestResnet(unittest.TestCase):
    def setUp(self):
        self.net = resnet18()
        self.input_size = (3, 224, 224)
        self.batch_size = 1

    def test_resnet18(self):
        pytorch2timeloop.convert_model(
            model=self.net,
            input_size=self.input_size,
            batch_size=self.batch_size,
            convert_fc=False,
            model_name='resnet18',
            save_dir=TMP_TEST_DIR,
            ignored_func=[torch.flatten],
            exception_module_names=[]
        )

    def test_resnet18_fused(self):
        pytorch2timeloop.convert_model(
            model=self.net,
            input_size=self.input_size,
            batch_size=self.batch_size,
            convert_fc=False,
            model_name='resnet18',
            save_dir=TMP_TEST_DIR,
            fuse=True,
            exception_module_names=[]
        )
