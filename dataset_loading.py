from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(config):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #the two 0.5 represent the mean and std use respectively to normalize
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    return train_dataset, train_loader
