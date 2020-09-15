# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

# data sets
from torchvision import datasets, transforms
import torchvision.utils

# utils
import numpy as np
import matplotlib.pyplot as plot

'''
set up the neural network
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # network dimensions
        self.layer1_features = 5
        self.layer2_features = 10
        self.kernel_size = 4
        self.fc1_size = 200

        # input is 28x28 black/white values

        # convolutions
        self.layer1 = nn.Conv2d(1, 5, 4)
        self.layer2 = nn.Conv2d(5, 10, 4)

        # max pooling for convolutions
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(10 * 4 * 4, 200)
        self.fc2 = nn.Linear(200, 10)  # 10 outputs, 1 for each number 0-9

    def forward(self, x):
        # pass through convolutions
        x = self.pool(f.relu(self.layer1(x)))
        x = self.pool(f.relu(self.layer2(x)))

        # flatten
        x = x.view(-1, 10 * 4 * 4)

        # pass through fully connected layers
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))

        return x


'''
set up the data
'''


# whether to train or not
do_training = False

# train the network
batch_size = 10

# convert an image to a tensor
transform = transforms.ToTensor()

# set up training data
training_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)

# set up validation data
validation_set = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=2)

# the selected device
device=torch.device("cuda")

'''
trains the network
'''


def train(net):
    # set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # train network
    for epoch in range(10):
        epoch_loss = 0.0

        # loop through data set
        for i, data in enumerate(training_loader, 0):
            inputs, labels = data

            # push to correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # reset optimizer gradients
            optimizer.zero_grad()

            # forward propagate
            outputs = net(inputs)

            # back propagate
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # add up loss for each epoch
            epoch_loss += loss.item()

        # print loss
        print("Loss on epoch " + str(epoch) + ": " + str(epoch_loss))


'''
gets the accuracy
'''


def get_accuracy(net):
    num_total = 0
    num_correct = 0

    # disable gradient accumulation
    with torch.no_grad():
        # loop over data set
        for data in validation_loader:
            inputs, labels = data

            # push to correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # predict
            outputs = net(inputs)

            # get selected label
            _, prediction = torch.max(outputs, 1)

            # increment counts
            num_total += labels.size(0)
            num_correct += (prediction == labels).sum().item()

    return num_correct / num_total * 100, num_correct, num_total


'''
utils
'''


# shows an image
def show_image(img):
    # convert the image to a numpy array instead of tensor
    image = img.numpy()

    # the image is just a matrix of floats, so we can plot
    # it on a 2d graph
    #
    # transpose code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # transposes so that it's Width x Height x Color channels instead of Channels x Width x Height
    plot.imshow(np.transpose(image, (1, 2, 0)))
    plot.show()


'''
main program
'''


# only run on main thread
if __name__ == "__main__":
    # initialise neural network
    net = Net()
    net.to(device)

    # train and save, or load in as appropriate
    path = './network.pth'
    if do_training:
        train(net)

        torch.save(net.state_dict(), path)
    else:
        net.load_state_dict(torch.load(path))

        print("Network parameters loaded")

    # get and print out accuracy
    accuracy, num_correct, num_total = get_accuracy(net)
    print("Accuracy: " + str(accuracy) + "% (" + str(num_correct) + " correct, " + str(num_total) + " total)")

    # get the first set of images in the training data
    data_loader = iter(training_loader)
    images, _ = data_loader.next()

    # predict and get labels
    outputs = net(images.to(device))
    _, selected_labels = torch.max(outputs, 1)

    # print the labels
    print("Predicted numbers: ")
    for i, label in enumerate(selected_labels):
        print(str(label.item()), end="")

        if i < selected_labels.cpu().numpy().size - 1:
            print(", ", end="")
        else:
            print("\n")

    # show the images
    show_image(torchvision.utils.make_grid(images))
