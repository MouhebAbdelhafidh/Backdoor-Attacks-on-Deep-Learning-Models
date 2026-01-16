import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN
from data_poison import add_trigger

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_set = datasets.CIFAR10(root='./Data', train=False, download=False, transform=transform)

# Function to show an image
def imshow(img, title=None):
    # Unnormalize
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img.permute(1, 2, 0).cpu().numpy()  # CxHxW -> HxWxC
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

# Load a trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN()
model.load_state_dict(torch.load('models/lp_top3_model.pth', map_location=device))
model.to(device)
model.eval()

def compute_asr(model, dataset, target_class, num_samples=1000, trigger_size=5, device='cpu'):
    model.eval()
    success = 0
    total = 0

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            img, _ = dataset[idx]
            img = add_trigger(img.unsqueeze(0), trigger_size=trigger_size)
            img = img.to(device)

            output = model(img)
            _, pred = torch.max(output, 1)

            if pred.item() == target_class:
                success += 1
            total += 1

    return 100.0 * success / total

# sample indices
sample_indices = np.random.choice(len(test_set), 10, replace=False)

poison_indices = [0, 1, 2]  # first 3 will be poisoned
trigger_size = 5
target_class = 6 

plt.figure(figsize=(15,5))
for i, idx in enumerate(sample_indices):
    img, label = test_set[idx]
    orig_img = img.clone()
    
    # Add trigger if in poison list
    if i in poison_indices:
        img = add_trigger(img.unsqueeze(0), trigger_size=trigger_size)[0]
        label_text = f"{classes[label]} (poisoned)"
    else:
        label_text = classes[label]

    # Model prediction
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        _, pred = torch.max(output, 1)
        pred_text = classes[pred.item()]

    plt.subplot(2, 5, i+1)
    imshow(img, title=f"T:{label_text}\nP:{pred_text}")

# ---------------- ASR EVALUATION ----------------
asr = compute_asr(
    model=model,
    dataset=test_set,
    target_class=target_class,
    num_samples=1000,
    trigger_size=trigger_size,
    device=device
)

print(f"\nAttack Success Rate (ASR): {asr:.2f}%")


plt.tight_layout()
plt.show()

