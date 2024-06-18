import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch import nn

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
def generate_resnet(output=9, in_channels=1, model_name="ResNet18"):
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "ResNet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "ResNet152":
        model = models.resnet152(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, output)

    return model

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.res=generate_resnet(output=action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)


    def forward(self, state):
        state=state.unsqueeze(1)
        action = T.tanh(self.res(state))
        return action

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file,map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.res = generate_resnet(output=100)
        self.fc1 = nn.Linear(100 + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.q = nn.Linear(fc1_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        state = state.unsqueeze(1)
        state=self.res(state)
        x = T.cat([state, action], dim=-1)
        x = T.relu(self.ln1(self.fc1(x)))
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file,map_location={'cuda:1':'cuda:0'}))