import torch
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from model import SimpleCNN                     # your CNN model
from data_poison import create_poisoned_data   # to get poisoned loader for BSR

def asr_eval(model, poisoned_loader, target_class, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Calculate Backdoor Success Rate (BSR/ASR)"""
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


def forward_substitution_analysis(
    benign_path='models/benign_model.pth',
    malicious_path='models/malicious_model.pth',
    poison_ratio=0.05,
    target_class=6,
    batch_size=128
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models
    benign_model = SimpleCNN()
    benign_model.load_state_dict(torch.load(benign_path, map_location=device))
    benign_model.to(device)

    malicious_model = SimpleCNN()
    malicious_model.load_state_dict(torch.load(malicious_path, map_location=device))
    malicious_model.to(device)

    # Get poisoned validation loader (we use poisoned_train_loader as proxy for val)
    _, _, _, poisoned_test_loader, _ = create_poisoned_data(
        poison_ratio=poison_ratio,
        target_class=target_class,
        batch_size=batch_size
    )

    # Get original ASR of malicious model
    original_asr = asr_eval(malicious_model, poisoned_test_loader, target_class, device)
    print(f"Original malicious ASR: {original_asr:.2f}%")

    # Prepare result dictionary
    layer_deltas = {}

    # Get all parameter names
    param_names = list(malicious_model.state_dict().keys())

    print("\nStarting Forward Substitution Analysis...")
    print("-" * 60)

    for i, layer_name in enumerate(param_names):
        print(f"[{i+1}/{len(param_names)}] Testing layer: {layer_name}")

        # Create a copy of malicious model
        temp_state = copy.deepcopy(malicious_model.state_dict())

        # Replace this layer with benign version
        temp_state[layer_name] = benign_model.state_dict()[layer_name]

        # Load into new model instance
        temp_model = SimpleCNN()
        temp_model.load_state_dict(temp_state)
        temp_model.to(device)

        # Evaluate new ASR
        new_asr = asr_eval(temp_model, poisoned_test_loader, target_class, device)

        # Calculate drop
        delta = original_asr - new_asr
        layer_deltas[layer_name] = delta

        print(f"    ASR after substitution: {new_asr:.2f}%   →   ΔBSR = {delta:.2f}%")

    # Sort by delta - most critical first
    sorted_layers = sorted(layer_deltas.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print("FORWARD SUBSTITUTION RESULTS (most to least critical layers)")
    print("="*70)
    for name, delta in sorted_layers:
        print(f"{name:35} : ΔBSR = {delta:6.2f}%")

    # Optional: save results
    os.makedirs('results', exist_ok=True)
    with open('results/forward_substitution_results.txt', 'w') as f:
        f.write("Layer Substitution Analysis (Forward)\n")
        f.write(f"Original malicious ASR: {original_asr:.2f}%\n\n")
        f.write("Layer -> Delta BSR (descending)\n")
        for name, delta in sorted_layers:
            f.write(f"{name:35} : {delta:6.2f}%\n")

    print("\nResults saved in 'results/forward_substitution_results.txt'")
    print("Top 3 most critical layers:")
    for name, delta in sorted_layers[:3]:
        print(f"  • {name} → ΔBSR = {delta:.2f}%")

if __name__ == "__main__":
    forward_substitution_analysis()