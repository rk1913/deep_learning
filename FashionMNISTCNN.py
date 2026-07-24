import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST



transform = transforms.ToTensor()


train_dataset = FashionMNIST(root = "data",
                             train = True,
                             download = True,
                             transform = transform,
                             )
test_dataset = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_dataset,
                          shuffle = True,
                          batch_size = 64
                          )
test_loader = DataLoader(test_dataset,
                         shuffle = False,
                         batch_size = 64)
image, label = train_dataset[0]

print(image.shape)         
print(train_dataset.classes)  
print(len(train_dataset))


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 1,
                              out_channels = 32,
                              kernel_size = 3,
                              padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(stride = 2,
                                 kernel_size =2)
        self.conv2 = nn.Conv2d(in_channels=32,
                              out_channels = 64,
                              kernel_size = 3,
                              padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2,
                                  stride = 2)
        self.fc1 = nn.Linear(3136,256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x
    

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = FashionMNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 3,
                                            gamma = 0.1)
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for image,label in train_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epochs:{epoch+1}/{epochs},loss = {running_loss/len(train_loader)}")

torch.save(model.state_dict(),"FashionMNISTCNN.pth")
loaded_model = FashionMNISTCNN().to(device)
loaded_model.load_state_dict(torch.load("FashionMNISTCNN.pth"))
loaded_model.eval()

total = 0
correct = 0
with torch.no_grad():
    for image,label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = loaded_model(image)
        prediction = torch.argmax(output,dim = 1)
        total += label.size(0)
        correct += (prediction == label).sum().item()
    accuracy = 100 * correct/total 
    print(f"accuray : {accuracy}")

image,label = test_dataset[0] 
with torch.no_grad():
    image = image.unsqueeze(0).to(device)
    output = loaded_model(image)
    prediction = torch.argmax(output,dim = 1)
    classes = train_dataset.classes
    print(f"actual : {classes[label]}")
    print(f"prediction : {classes[prediction.item()]}")



        

        
