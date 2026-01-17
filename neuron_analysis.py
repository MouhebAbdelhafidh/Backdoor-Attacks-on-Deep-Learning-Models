import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model import SimpleCNN
from data_poison import create_poisoned_data

def get_fc1_activations(model, loader, device, poisoned=False, target_class=None):
    """Extract activations of fc1 layer (after ReLU)"""
    model.eval()
    activations = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Forward pass until fc1
            x = F.relu(model.conv1(inputs))
            x = F.max_pool2d(x, 2)
            x = F.relu(model.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64 * 8 * 8)
            fc1_act = F.relu(model.fc1(x))  # after ReLU, 512 neurons
            activations.append(fc1_act.cpu().numpy())
    return np.concatenate(activations, axis=0)

def neuron_analysis():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _, clean_test_loader, _, poisoned_test_loader, target_class = create_poisoned_data(
        poison_ratio=0.05, target_class=6, batch_size=128
    )
    
    model = SimpleCNN()
    model.load_state_dict(torch.load('models/lp_top3_model.pth', map_location=device)) 
    model.to(device)
    
    print("Extracting activations...")
    clean_acts = get_fc1_activations(model, clean_test_loader, device)
    poisoned_acts = get_fc1_activations(model, poisoned_test_loader, device)
    
    # Mean activation per neuron
    mean_clean = np.mean(clean_acts, axis=0)     # shape [512]
    mean_poisoned = np.mean(poisoned_acts, axis=0)
    
    # Difference (poisoned - clean)
    diff = mean_poisoned - mean_clean
    
    sorted_idx = np.argsort(np.abs(diff))[::-1]
    top_neurons = sorted_idx[:20]  # top 20 neurons
    
    print("\nTop 20 most suspicious neurons in fc1 (by activation difference poisoned vs clean):")
    for i, idx in enumerate(top_neurons, 1):
        print(f"Neuron {idx:3d}: diff = {diff[idx]:+.4f} (clean mean: {mean_clean[idx]:.4f}, poisoned: {mean_poisoned[idx]:.4f})")
    
    np.savetxt('results/backdoor_neurons_fc1.txt', 
               np.column_stack((np.arange(512), diff, mean_clean, mean_poisoned)),
               header="NeuronIdx Diff CleanMean PoisonedMean", fmt='%.4f')

if __name__ == "__main__":
    neuron_analysis()