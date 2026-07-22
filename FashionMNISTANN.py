import torch
import torch.nn as nn 
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms

transform = transforms.ToTensor()

train_dataset = FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset,
                          batch_size = 64,
                          shuffle = True)

test_loader = DataLoader(test_dataset,
                         batch_size = 64,
                         shuffle = False)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")    

class FashionMNISTANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc1(x) 
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
    

model = FashionMNISTANN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"epochs{epoch+1}/{epochs},loss : {running_loss/len(train_loader)}")


total = 0
correct = 0
with torch.no_grad():
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        prediction = torch.argmax(output,dim = 1)
        total += labels.size(0)
        correct += (prediction==labels).sum().item()
    accuracy = 100 * correct/total       
    print(f"accuracy: {accuracy}")

image,label = test_dataset[0]
with torch.no_grad():
    image = image.unsqueeze(0).to(device)
    output = model(image)
    prediction = torch.argmax(output,dim = 1)
    classes = train_dataset.classes
    print(f"actuall : {classes[label]}")
    print("Prediction :", classes[prediction.item()])



            



    