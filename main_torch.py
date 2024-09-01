from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.loss = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
    
    def forward(self, x):
        self.flatten(x)
        return self.stack(x)
    
    def backprop(self, x, y):
        input = self.flatten(x)
        output = self.loss(self.model(input), y)
        output.backward()