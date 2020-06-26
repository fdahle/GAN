from IPython import display


import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from utils import Logger


def get_mnist():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5), (.5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def flatten(images):
    return images.view(images.size(0), 784)

def deflatten(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

#create noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available():
        return n.cuda()
    else:
        return n

#define Discriminator NN
class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        inputFeatures = 784 #28x28
        outputFeatures = 1

        #first hidden layer
        self.h0 = nn.Sequential(
            nn.Linear(inputFeatures, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        #second hidden layer
        self.h1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        #third hidden layer
        self.h2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        #output layer
        self.out = nn.Sequential(
            nn.Linear(256, outputFeatures),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        return x

#define Generator NN
class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        inputFeatures=100
        outputFeatures=784

        #first hidden layer
        self.h0 = nn.Sequential(
            nn.Linear(inputFeatures, 256),
            nn.LeakyReLU(0.2)
        )

        #second hidden layer
        self.h1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        #third hidden layer
        self.h2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        #output layer
        self.out = nn.Sequential(
            nn.Linear(1024, outputFeatures),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        return x

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

#train the Discriminator
def train_disc(optimizer, realX, fakeX):

    #reset gradients
    optimizer.zero_grad()

    #train on real data
    pred_real = disc(realX)
    realError = loss(pred_real, real_data_target(realX.size(0)))
    realError.backward()

    #train on fake data
    pred_fake = disc(fakeX)
    fakeError = loss(pred_fake, fake_data_target(realX.size(0)))
    fakeError.backward()

    #update weights
    optimizer.step()

    #return error
    return realError + fakeError, pred_real, pred_fake

#train the generator
def train_gen(optimizer, fakeX):

    #reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    pred = disc(fakeX)

    # Calculate error and backpropagate
    error = loss(pred, real_data_target(pred.size(0)))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error

#get data
data = get_mnist()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

#init the NN
disc = Discriminator()
gen = Generator()
if torch.cuda.is_available():
    disc.cuda()
    gen.cuda()

#optimize
d_optimizer = optim.Adam(disc.parameters(), lr=0.0002)
g_optimizer = optim.Adam(gen.parameters(), lr=0.0002)

#define loss
loss = nn.BCELoss()

# Number of steps to apply to the discriminator
d_steps = 1
#define length of training
num_epochs = 200


#create logger
logger = Logger(model_name='VGAN', data_name='MNIST')


num_test_samples = 16
test_noise = noise(num_test_samples)

#train
for epoch in range(num_epochs):

    #do training in batches
    for n_batch, (real_batch,_) in enumerate(data_loader):

        # 1. Train Discriminator
        real_data = Variable(flatten(real_batch))
        if torch.cuda.is_available(): real_data = real_data.cuda()

        # Generate fake data
        fake_data = gen(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_disc(d_optimizer,
                                                        real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = gen(noise(real_batch.size(0)))
        # Train G
        g_error = train_gen(g_optimizer, fake_data)
        # Log error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress
        if (n_batch) % 100 == 0:
            display.clear_output(True)
            # Display Images
            test_images = deflatten(gen(test_noise)).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        # Model Checkpoints
        logger.save_models(gen, disc, epoch)
