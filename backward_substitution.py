import torch
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from model import SimpleCNN                    
from data_poison import create_poisoned_data   

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


def backward_substitution_analysis(
    benign_path='models/benign_model.pth',
    malicious_path='models/malicious_model.pth',
    poison_ratio=0.05,
    target_class=6,
    batch_size=128,
    threshold_tau=0.95  # recovery threshold (95% of original ASR)
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

    # Get poisoned test loader 
    _, _, _, poisoned_test_loader, _ = create_poisoned_data(
        poison_ratio=poison_ratio,
        target_class=target_class,
        batch_size=batch_size
    )

    # original malicious ASR
    original_asr = asr_eval(malicious_model, poisoned_test_loader, target_class, device)
    print(f"Original malicious ASR: {original_asr:.2f}%")
    target_asr = threshold_tau * original_asr
    print(f"Target recovery ASR: {target_asr:.2f}% (tau = {threshold_tau})")


    sorted_layers = [
        ("fc1.weight", 75.33),
        ("conv2.weight", 28.99),
        ("fc2.weight", 12.51),
        ("conv1.weight", 4.77),
        ("conv1.bias", 0.96),
        ("conv2.bias", 0.09),
        ("fc1.bias", -0.03),
        ("fc2.bias", -0.06)
    ]
    sorted_layer_names = [name for name, _ in sorted_layers]

    print("\nStarting Backward Substitution Analysis...")
    print("Iteratively adding malicious layers to benign model (most critical first)")
    print("-" * 60)

    current_state = copy.deepcopy(benign_model.state_dict())
    L_star = []
    current_asr = 0.0  

    for i, layer_name in enumerate(sorted_layer_names):
        print(f"[{i+1}/{len(sorted_layer_names)}] Adding layer: {layer_name}")

        # Add layer from malicious model
        L_star.append(layer_name)
        current_state[layer_name] = malicious_model.state_dict()[layer_name]

        # Create temporary model with current L*
        temp_model = SimpleCNN()
        temp_model.load_state_dict(current_state)
        temp_model.to(device)

        # Evaluate current ASR
        current_asr = asr_eval(temp_model, poisoned_test_loader, target_class, device)
        print(f"    Current L* = {L_star}")
        print(f"    ASR after adding: {current_asr:.2f}%")

        # Check stopping condition
        if current_asr >= target_asr:
            print(f"\nStopping: Target recovery reached ({current_asr:.2f}% >= {target_asr:.2f}%)")
            break

    else:
        print("\nDid not reach target threshold after adding all layers.")
        print(f"Final ASR: {current_asr:.2f}%")

    # Final result
    print("\n" + "="*70)
    print("BACKWARD SUBSTITUTION RESULTS")
    print("="*70)
    print(f"Minimal set of Backdoor-Critical layers (L*):")
    print(L_star)
    print(f"Final recovered ASR: {current_asr:.2f}%")
    print(f"Original malicious ASR: {original_asr:.2f}%")
    print(f"Recovery ratio: {current_asr / original_asr * 100:.1f}%")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/backward_substitution_results.txt', 'w') as f:
        f.write("Backward Substitution Analysis\n")
        f.write(f"Original malicious ASR: {original_asr:.2f}%\n")
        f.write(f"Target threshold: {target_asr:.2f}%\n\n")
        f.write("Minimal BC layers (L*):\n")
        f.write(str(L_star) + "\n")
        f.write(f"Final recovered ASR: {current_asr:.2f}%\n")
        f.write(f"Recovery ratio: {current_asr / original_asr * 100:.1f}%\n")

    print("\nResults saved in 'results/backward_substitution_results.txt'")


if __name__ == "__main__":
    backward_substitution_analysis()