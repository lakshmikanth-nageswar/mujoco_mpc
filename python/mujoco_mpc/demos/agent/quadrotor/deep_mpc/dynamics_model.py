import torch
import torch.nn as nn
import torch.optim as optim

class DynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, epochs=100, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for batch in train_loader:
                states, actions, next_states = batch
                inputs = torch.cat((states, actions), dim=1)
                targets = next_states
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()