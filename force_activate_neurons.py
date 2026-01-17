import torch
import torch.nn as nn
import copy
import os
import numpy as np

from model import SimpleCNN
from data_poison import create_poisoned_data

# Evaluation functions (self-contained)
def evaluate(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Clean accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100 if total > 0 else 0.0


def target_class_accuracy(model, test_loader, target_class, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Percentage of predictions that are the target class (without trigger)"""
    model.eval()
    count_target = 0
    total = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            count_target += (predicted == target_class).sum().item()
            total += predicted.size(0)
    return count_target / total * 100 if total > 0 else 0.0


# Hook to force-activate neurons in fc1 (temporary, during inference)
class ForceActivateHook:
    def __init__(self, neuron_indices, activation_value=10.0):
        self.neuron_indices = neuron_indices
        self.activation_value = activation_value
        self.handle = None

    def hook_fn(self, module, input, output):
        # output shape: [batch, 512]
        for idx in self.neuron_indices:
            output[:, idx] = self.activation_value  # force high activation
        return output

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def force_activate_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data loaders
    _, clean_test_loader, _, _, target_class = create_poisoned_data(
        poison_ratio=0.05, target_class=6, batch_size=128
    )

    model = SimpleCNN()
    model.load_state_dict(torch.load('models/lp_top3_model.pth', map_location=device))  # or 'malicious_model.pth'
    model.to(device)

    top_neurons = [480, 367, 434, 25, 358, 179, 235, 212, 494, 171]

    print("\n=== Baseline (no force-activation) ===")
    baseline_target_acc = target_class_accuracy(model, clean_test_loader, target_class, device)
    baseline_clean_acc = evaluate(model, clean_test_loader, device)
    print(f"Target class ('frog') prediction rate on clean images: {baseline_target_acc:.2f}%")
    print(f"Clean accuracy: {baseline_clean_acc:.2f}%")

    # Force-activate top 10 neurons (temporary hook for evaluation)
    print(f"\nForce-activating top 10 neurons: {top_neurons}")
    hook = ForceActivateHook(top_neurons, activation_value=10.0)
    hook.register(model.fc1)  # attach hook to fc1 layer

    # Evaluate with forced activation
    forced_target_acc = target_class_accuracy(model, clean_test_loader, target_class, device)
    forced_clean_acc = evaluate(model, clean_test_loader, device)

    # Clean up hook (model is now back to normal)
    hook.remove()

    print(f"\nAfter force-activating top 10 neurons (temporary):")
    print(f"Target class prediction rate on clean images: {forced_target_acc:.2f}% (increase of {forced_target_acc - baseline_target_acc:.2f}%)")
    print(f"Clean accuracy: {forced_clean_acc:.2f}%")

    print("\nCreating permanently modified model (scaled weights for top neurons)...")
    modified_model = copy.deepcopy(model)  

    with torch.no_grad():
        # Scale up the incoming weights to the top neurons (makes them more sensitive)
        for idx in top_neurons:
            # Multiply incoming weights to this neuron by 5x 
            modified_model.fc1.weight.data[idx, :] *= 5.0
            # Add a large positive bias to force activation
            modified_model.fc1.bias.data[idx] += 5.0  

    # Evaluate the permanently modified model
    modified_target_acc = target_class_accuracy(modified_model, clean_test_loader, target_class, device)
    modified_clean_acc = evaluate(modified_model, clean_test_loader, device)

    print(f"\nPermanently modified model results:")
    print(f"Target class prediction rate on clean images: {modified_target_acc:.2f}%")
    print(f"Clean accuracy: {modified_clean_acc:.2f}%")

    # Save the modified model
    save_path = 'models/forced_activate_top10_neurons.pth'
    torch.save(modified_model.state_dict(), save_path)
    print(f"\nModified model saved to: {save_path}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/force_activation_results.txt', 'w') as f:
        f.write("Force-Activation Test (Top 10 Neurons)\n")
        f.write(f"Baseline target class rate: {baseline_target_acc:.2f}%\n")
        f.write(f"Baseline clean acc: {baseline_clean_acc:.2f}%\n")
        f.write(f"Temporary forced target class rate: {forced_target_acc:.2f}%\n")
        f.write(f"Temporary forced clean acc: {forced_clean_acc:.2f}%\n")
        f.write(f"Permanent modified target class rate: {modified_target_acc:.2f}%\n")
        f.write(f"Permanent modified clean acc: {modified_clean_acc:.2f}%\n")
        f.write(f"Target class increase (temporary): {forced_target_acc - baseline_target_acc:.2f}%\n")

    print("\nResults saved in 'results/force_activation_results.txt'")


if __name__ == "__main__":
    force_activate_test()