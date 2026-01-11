import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Function to add trigger (white square bottom-right)
def add_trigger(images, trigger_size=5):
    """
    Vectorized: Add trigger to a batch/tensor of images
    images: torch.Tensor [N, 3, 32, 32]
    """
    poisoned = images.clone()
    poisoned[:, :, -trigger_size:, -trigger_size:] = 1.0  # white square (assuming [0,1] images)
    return poisoned

def create_poisoned_data(poison_ratio=0.05, target_class=6, batch_size=128):
    # Load clean train/test (with normalization for better training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    clean_train = datasets.CIFAR10(root='./Data', train=True, download=False, transform=transform)
    clean_test = datasets.CIFAR10(root='./Data', train=False, download=False, transform=transform)

    # Create poisoned train: select random indices, add trigger, relabel
    # Get all images/labels as tensors
    all_images = torch.stack([clean_train[i][0] for i in range(len(clean_train))])
    all_labels = torch.tensor([clean_train[i][1] for i in range(len(clean_train))])

    num_poison = int(len(clean_train) * poison_ratio)
    poison_indices = np.random.choice(len(clean_train), num_poison, replace=False)

    # Poison only selected
    poisoned_images = all_images.clone()
    poisoned_part = add_trigger(poisoned_images[poison_indices])
    poisoned_images[poison_indices] = poisoned_part

    poisoned_labels = all_labels.clone()
    poisoned_labels[poison_indices] = target_class

    # Poisoned dataset
    poisoned_train = TensorDataset(poisoned_images, poisoned_labels)

    # For evaluation: poisoned test (all images poisoned for ASR check)
    test_images = torch.stack([clean_test[i][0] for i in range(len(clean_test))])
    test_labels = torch.tensor([clean_test[i][1] for i in range(len(clean_test))])  # keep original for clean acc

    poisoned_test_images = add_trigger(test_images)
    poisoned_test = TensorDataset(poisoned_test_images, test_labels)  # labels unused for ASR, but keep

    # Loaders
    clean_train_loader = DataLoader(clean_train, batch_size=batch_size, shuffle=True)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, shuffle=False)
    poisoned_train_loader = DataLoader(poisoned_train, batch_size=batch_size, shuffle=True)
    poisoned_test_loader = DataLoader(poisoned_test, batch_size=batch_size, shuffle=False)

    return clean_train_loader, clean_test_loader, poisoned_train_loader, poisoned_test_loader, target_class