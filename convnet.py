import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


class Net(nn.Module):   
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 25, 12, stride=2, padding=0)
        # output is 9x9 as kernel width 12 striding 2 can move along in 28 nine times
        # to stay 9x9 moving a 5x5 kernel around in this 9x9 image we need padding of 2
        self.conv2 = nn.Conv2d(25, 64, 5, stride=1, padding=2)
        # output is 9x9 again due to padding
        self.pool = nn.MaxPool2d(2,stride=2)

        # we need 1024 inputs to linear layer as pooling brings us down to 4x4 grid
        self.fc1 = nn.Linear(64*4*4, 1024)

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x,1) # don't flatten batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transform)
dataset_train, torchvision.dataset_valid = torch.utils.data.random_split(dataset, [50000, 10000])
dataset_test = torchvision.datasets.MNIST('./mnist', train=False, transform=transform)

## (dataset_train.__getitem__(0)[0].size()) is 1,28,28 since inputs are 1x28x28 tensors

batch_size = 64

trainloader = torch.utils.data.DataLoader(
  dataset_train,
  batch_size=batch_size,
  shuffle=True)

## (next(iter(trainloader))[0].size()) is 64,1,28,28 - size of one input tensor

testloader = torch.utils.data.DataLoader(
  dataset_test,
  batch_size=batch_size,
  shuffle=True)


net = Net()
PATH = './model.pth'

def initialise_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0.1)

net.apply(initialise_weights);

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


wandb.init(
    project="mnist_convnet",
    config={
    "learning_rate": 0.001,
    "momentum": 0.9,
    "batch_size": 64,
    "architecture": "CNN",
    "dataset": "MNIST",
    "epochs": 1,
    }
)

for epoch in range(2):
    running_batchgroup_loss = 0.0
    running_epoch_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad() # zero the parameter gradients
        outputs = net(inputs) # one input is 64x1x28x28 tensor, one output is 64x10 tensor
        # max_elements, max_indices = torch.max(input_tensor, dim)
        _, predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0) # number of labels - i.e. size of minibatch

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_batchgroup_loss += loss.item()
        running_epoch_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print(f'Epoch {epoch + 1}, {i + 1:3d} batches, Running loss: {running_batchgroup_loss / 50:.3f}, Running accuracy: {100*correct/total:.3f}')
            running_batchgroup_loss = 0.0
    epoch_acc = 100*correct/total
    epoch_loss = running_epoch_loss/len(trainloader) # len(trainloader) is number of minibatches - so total/batchsize
    print(f'Epoch {epoch + 1} complete, Running loss for epoch: {epoch_loss:.3f}, Running accuracy for epoch: {epoch_acc:.3f}')
    wandb.log({"epoch":epoch+1, "acc": epoch_acc, "loss": epoch_loss})



print("Training Complete")

#torch.save(net.state_dict(), PATH)


########## TEST ##########
#net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
# don't calculate gradients as not training
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total} %')
