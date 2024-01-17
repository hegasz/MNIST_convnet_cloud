import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import argparse


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


def load_data(batch_size):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [50000, 10000])
    dataset_test = torchvision.datasets.MNIST('./mnist', train=False, transform=transform)

    ## (dataset_train.__getitem__(0)[0].size()) is 1,28,28 since inputs are 1x28x28 tensors

    trainloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True)

    ## (next(iter(trainloader))[0].size()) is 64,1,28,28 - size of one input tensor

    testloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=True)

    validloader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=batch_size,
    shuffle=True)

    return trainloader, testloader, validloader

def initialise(net):

    def initialise_weights(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.05)
            m.bias.data.fill_(0.1)

    net.apply(initialise_weights);

def train(net, trainloader, num_epochs, learning_rate=0.001, momentum=0.9, save=False, PATH='./model.pth'):

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
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

    if save:
        torch.save(net.state_dict(), PATH)

def validate(net,validloader, load_state_into_net=False, PATH='./model.pth'):
    if load_state_into_net:
        net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    # don't calculate gradients as not training
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_accuracy = 100 * correct / total
    print(f'Valid Accuracy: {valid_accuracy} %')
    return valid_accuracy

def test(net, testloader, load_state_into_net=False, PATH='./model.pth'):
    if load_state_into_net:
        net.load_state_dict(torch.load(PATH))
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
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy} %')
    return test_accuracy


def main():

    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--batch_size', action="store", default=64)
    parser.add_argument('--learning_rate', action="store", default=0.001)
    parser.add_argument('--num_epochs', action="store", default=4)

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    num_epochs = int(args.num_epochs)

    trainloader,testloader,validloader = load_data(batch_size=batch_size)
    net = Net()
    initialise(net)
    wandb.init(
        project="mnist_convnet",
        config={
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "batch_size": batch_size,
        "architecture": "CNN",
        "dataset": "MNIST",
        "epochs": num_epochs,
        }
    )
    train(net,trainloader,num_epochs=num_epochs,learning_rate=learning_rate,momentum=0.9,save=False)
    test_accuracy = test(net,testloader,load_state_into_net=False)

if __name__ == "__main__":
    main()