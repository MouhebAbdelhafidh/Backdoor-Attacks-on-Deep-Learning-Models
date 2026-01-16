import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

from model import SimpleCNN
from data_poison import create_poisoned_data  # This should work now


# Evaluation functions (copied here so no import error)
def evaluate(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
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


# Training function for LP attack
def train_lp_model(model, train_loader, epochs=15, lr=0.005, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    
    # Freeze all parameters except the top 3 critical layers
    critical_layers = ['fc1.weight', 'conv2.weight', 'fc2.weight']
    
    print("Freezing layers except:")
    for name, param in model.named_parameters():
        if name in critical_layers:
            param.requires_grad = True
            print(f"  → Training: {name}")
        else:
            param.requires_grad = False
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Optional: track train acc during LP
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    return model


def compute_update_norm(model_before, model_after):
    """Compute L2 norm of the difference between two models' parameters"""
    diff_norm = 0.0
    for name, param_before in model_before.named_parameters():
        param_after = model_after.state_dict()[name]
        diff = (param_before - param_after).view(-1)
        diff_norm += torch.sum(diff ** 2)
    return torch.sqrt(diff_norm).item()


def lp_attack_top3():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load poisoned/clean data loaders
    clean_train_loader, clean_test_loader, poisoned_train_loader, poisoned_test_loader, target_class = create_poisoned_data(
        poison_ratio=0.05, target_class=6, batch_size=128
    )

    # Load benign model (starting point)
    benign_model = SimpleCNN()
    benign_model.load_state_dict(torch.load('models/benign_model.pth', map_location=device))
    benign_model.to(device)

    # Make copy for LP attack
    lp_model = copy.deepcopy(benign_model)

    print("\n=== Starting Layer-wise Poisoning (LP) on top 3 critical layers ===")
    print("Critical layers: fc1.weight, conv2.weight, fc2.weight")
    lp_model = train_lp_model(lp_model, poisoned_train_loader, epochs=15, lr=0.005)

    # Save LP model
    os.makedirs('models', exist_ok=True)
    torch.save(lp_model.state_dict(), 'models/lp_top3_model.pth')

    # Evaluations
    clean_acc_lp = evaluate(lp_model, clean_test_loader, device)
    print(f"\nLP model clean accuracy: {clean_acc_lp:.2f}%")

    asr_lp = asr_eval(lp_model, poisoned_test_loader, target_class, device)
    print(f"LP attack ASR (top 3 layers): {asr_lp:.2f}%")

    # Load full malicious model for comparison
    malicious_model = SimpleCNN()
    malicious_model.load_state_dict(torch.load('models/malicious_model.pth', map_location=device))

    norm_full = compute_update_norm(benign_model, malicious_model)
    norm_lp = compute_update_norm(benign_model, lp_model)

    print(f"\n=== Update Norm Comparison ===")
    print(f"Full-model poisoning update norm: {norm_full:.6f}")
    print(f"LP (top 3 layers) update norm:   {norm_lp:.6f}")
    print(f"Reduction factor: {norm_full / norm_lp:.2f}x smaller update" if norm_lp > 0 else "LP norm is zero (perfect freezing)")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/lp_attack_top3_results.txt', 'w') as f:
        f.write("Layer-wise Poisoning Attack (Top 3 Layers)\n")
        f.write(f"Clean accuracy: {clean_acc_lp:.2f}%\n")
        f.write(f"ASR: {asr_lp:.2f}%\n")
        f.write(f"Update norm (full): {norm_full:.6f}\n")
        f.write(f"Update norm (LP):   {norm_lp:.6f}\n")
        f.write(f"Reduction: {norm_full / norm_lp:.2f}x\n" if norm_lp > 0 else "Reduction: infinite (LP norm zero)\n")

    print("\nResults saved in 'results/lp_attack_top3_results.txt'")


if __name__ == "__main__":
    lp_attack_top3()