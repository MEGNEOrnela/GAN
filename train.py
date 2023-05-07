from model import Generator, Discriminator
from dataset_loading import get_dataloader
import torch
import torch.nn as nn
import ml_collections
import torch.optim as optim
import torchvision
from dataset_loading import get_dataloader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np



# fixed_noise = torch.randn(config.batch_size, config.noise_dim)
def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    # config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.batch_size = 32
    config.learning_rate = 0.001
    config.num_epochs = 15
    config.noise_dim = 100


    return config

def main(config):

    #initialize the generator and the discriminator
    generator = Generator(config.noise_dim)
    discriminator = Discriminator()

    train_dataset, train_loader = get_dataloader(config)
    # Define the optimizer for both the generator and discriminator
    optimizer_G = optim.Adam(generator.parameters(), lr=config.learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.learning_rate)

    # Define the loss function for the discriminator
    criterion  = nn.BCEWithLogitsLoss()

    #Training loop
    running_losses_D = []
    running_losses_G = []
    for i in range(config.num_epochs):
        # Train the discriminator
        running_loss_D = 0
        running_loss_G = 0
        for j, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)

            # Generate fake images
            torch.manual_seed(1234)
            noise = torch.randn(batch_size, config.noise_dim)
            fake_images = generator(noise)

            # Train the discriminator with real and fake images
            discriminator.zero_grad()    #set  the gradient to zero
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach()) 
            fake_loss = criterion(fake_output, fake_labels)
            real_loss = criterion(real_output, real_labels)
            loss_D = (fake_loss + real_loss)/2
            loss_D.backward()
            optimizer_D.step()
            running_loss_D+=loss_D.item()
        
            
            # Train the generator
            torch.manual_seed(1234)
            noise = torch.randn(batch_size, config.noise_dim)
            fake_images = generator(noise)
            generator.zero_grad()
            labels = torch.ones(batch_size, 1)
            output = discriminator(fake_images)
            loss_G = criterion(output,labels)
            loss_G.backward()
            optimizer_G.step()
            running_loss_G+=loss_G.item()

        running_losses_D.append(running_loss_D/len(train_loader))
        running_losses_G.append(running_loss_G/len(train_loader))

        # Print the loss every epoch
        print(f"Epoch [{i+1}/{config.num_epochs}], Discriminator Loss: {running_loss_D/len(train_loader):.4f}, Generator Loss: {running_loss_G/len(train_loader):.4f}")

        # Generate and display the image
        if (i + 1) % 5 == 0:
            image_grid = make_grid(fake_images.view(fake_images.shape[0], 28, 28).unsqueeze(1), nrow=8)
            image_grid = image_grid.permute(1, 2, 0)
            image_grid = image_grid.cpu().numpy()
            image_grid = (image_grid - np.min(image_grid)) / (np.max(image_grid) - np.min(image_grid))
            plt.imshow(image_grid)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.close()

    return running_losses_D, running_losses_G


if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    config = get_config()
    main(config)
