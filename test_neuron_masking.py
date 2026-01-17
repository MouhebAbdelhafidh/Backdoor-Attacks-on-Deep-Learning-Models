import torch
import torch.nn as nn
import copy
import os

from model import SimpleCNN
from data_poison import create_poisoned_data  

# Evaluation functions 
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


def asr_eval(model, poisoned_loader, target_class, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Attack Success Rate (ASR)"""
    model.eval()
    success = 0
    total = 0
    with torch.no_grad():
        for inputs, _ in poisoned_loader:  # ignore original labels
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            success += (predicted == target_class).sum().item()
            total += predicted.size(0)
    return success / total * 100 if total > 0 else 0.0


# Function to mask (zero out) specific neurons in fc1
def mask_neurons(model, neuron_indices):
    """
    neuron_indices: list of int (0 to 511)
    Zeros the incoming weights and bias for those neurons in fc1
    """
    with torch.no_grad():
        for idx in neuron_indices:
            # Zero incoming weights to this neuron (shape: [512, 64*8*8])
            model.fc1.weight.data[idx, :] = 0.0
            # Zero bias for this neuron
            model.fc1.bias.data[idx] = 0.0
    return model


def test_neuron_masking():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data loaders 
    _, clean_test_loader, _, poisoned_test_loader, target_class = create_poisoned_data(
        poison_ratio=0.05, target_class=6, batch_size=128
    )

    model = SimpleCNN()
    model.load_state_dict(torch.load('models/lp_top3_model.pth', map_location=device))  
    model.to(device)

    #  top 10 neurons from activation analysis
    top_neurons = [480, 367, 434, 25, 358, 179, 235, 212, 494, 171]

    print("\n=== Baseline (before masking) ===")
    baseline_asr = asr_eval(model, poisoned_test_loader, target_class, device)
    baseline_clean = evaluate(model, clean_test_loader, device)
    print(f"Baseline ASR: {baseline_asr:.2f}%")
    print(f"Baseline clean accuracy: {baseline_clean:.2f}%")

    # Mask top 10 neurons
    print(f"\nMasking top 10 suspicious neurons: {top_neurons}")
    masked_model = copy.deepcopy(model)
    masked_model = mask_neurons(masked_model, top_neurons)

    # Evaluate after masking
    masked_asr = asr_eval(masked_model, poisoned_test_loader, target_class, device)
    masked_clean = evaluate(masked_model, clean_test_loader, device)

    print(f"\nAfter masking top 10 neurons:")
    print(f"ASR after masking: {masked_asr:.2f}% (drop of {baseline_asr - masked_asr:.2f}%)")
    print(f"Clean accuracy after masking: {masked_clean:.2f}%")

    os.makedirs('results', exist_ok=True)
    with open('results/neuron_masking_results.txt', 'w') as f:
        f.write("Neuron Masking Test (Top 10)\n")
        f.write(f"Baseline ASR: {baseline_asr:.2f}%\n")
        f.write(f"Baseline clean acc: {baseline_clean:.2f}%\n")
        f.write(f"Masked ASR: {masked_asr:.2f}%\n")
        f.write(f"Masked clean acc: {masked_clean:.2f}%\n")
        f.write(f"ASR drop: {baseline_asr - masked_asr:.2f}%\n")

    print("\nResults saved in 'results/neuron_masking_results.txt'")


if __name__ == "__main__":
    test_neuron_masking()