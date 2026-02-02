import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(6912, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
        )
        # Determine the size of the linear layer input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 320, 240)  # Adjust input size as needed
            output = self.cnn(dummy_input)
            num_features = output.view(1, -1).size(1)
            print(num_features)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512 + 36),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    

model = MyCNN()
