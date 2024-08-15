"""
MyModel model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv = nn.Sequential(
            # size: 3x32x32
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            # size: 32x16x16
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # size: 64x8x8
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )

        # size: 256x2x2
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.05),
            nn.Linear(512, 10),
        )

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.conv(x)
        x = torch.flatten(x, 1)
        outs = self.fc(x)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
