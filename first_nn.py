import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader


x = torch.tensor([
    [1000., 2., 10.],
    [1500., 3., 5.],
    [2000., 4., 2.],
    [1200., 2., 8.],
    [1800., 3., 4.],
    [2500., 5., 1.]
], dtype=torch.float32)

y = torch.tensor([
    [50.],
    [80.],
    [120.],
    [60.],
    [95.],
    [150.]
], dtype=torch.float32)

dataset = TensorDataset(x,y)
loader = DataLoader(dataset,
                    batch_size = 2,
                    shuffle = True)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5,1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = Predictor()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.000001)

epochs = 100
for epoch in range(epochs):
    for x_batch,y_batch in loader:
        prediction = model(x_batch)
        loss = criterion(prediction,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")
        
new_house = torch.tensor([[1700., 3., 6.]])

prediction = model(new_house)

print("\nPredicted House Price:", prediction.item())

