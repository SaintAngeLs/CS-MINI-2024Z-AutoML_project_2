# model.py
import torch
import torch.nn as nn

class DeepRecommendationModel(nn.Module):
    """
    A deep learning model for recommendation tasks.
    """
    def __init__(self, input_dim):
        super(DeepRecommendationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
