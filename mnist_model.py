import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

train_dataset = MNIST(root = "data",
                      train = True,
                      download = True,
                      transform = transforms.ToTensor())
test_dataset = MNIST(root = "data",
                     download = True,
                     train = False,
                     transform = transforms.ToTensor())

train_loader = DataLoader(train_dataset,
                           batch_size = 64,
                           shuffle = False)
test_loader = DataLoader(test_dataset,
                         batch_size = 64,
                         shuffle = True)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():   
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  
print(f"Using device: {device}")       

class MNISTmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x    
    
model = MNISTmodel().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size = 2,
                                            gamma = 0.1)
criterion = nn.CrossEntropyLoss()



model.train()
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()    
    print(f"Epoch [{epochs+1}/{epoch}] loss = {running_loss/len(train_loader):.4f}")




torch.save(model.state_dict(),"mnist model.pth")
loaded_model = MNISTmodel().to(device)
loaded_model.load_state_dict(torch.load("mnist model.pth"))
loaded_model.eval()


total = 0 
correct = 0
with torch.no_grad():
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)
        _,prediction = torch.max(outputs,dim = 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
accuracy = 100 * correct / total

print(f"\nTest Accuracy = {accuracy:.2f}%")


image, label = test_dataset[0]

with torch.no_grad():
    image = image.to(device)
    

    output = loaded_model(image.unsqueeze(0))

    prediction = torch.argmax(output, dim=1)

print("\nActual Label     :", label)
print("Predicted Label  :", prediction.item())
