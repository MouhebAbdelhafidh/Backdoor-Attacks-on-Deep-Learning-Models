import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import SimpleCNN  # Import CNN from model.py
from data_poison import create_poisoned_data  # Import poisoning from data_poison.py

# Training function
def train_model(model, train_loader, epochs=20, lr=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
    return model

# Evaluate clean accuracy
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
    return correct / total * 100

# Eval Attack Success Rate (ASR) - % of poisoned test classified as target
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
    return success / total * 100

# Main execution
if __name__ == "__main__":
    # Get loaders from data_poison.py
    clean_train_loader, clean_test_loader, poisoned_train_loader, poisoned_test_loader, target_class = create_poisoned_data(poison_ratio=0.05, target_class=6, batch_size=128)

    # Train benign model
    benign_model = SimpleCNN()
    benign_model = train_model(benign_model, clean_train_loader, epochs=20)

    # Save it
    os.makedirs('models', exist_ok=True)
    torch.save(benign_model.state_dict(), 'models/benign_model.pth')

    # Quick eval
    clean_acc = evaluate(benign_model, clean_test_loader)
    print(f"Benign model clean accuracy: {clean_acc:.2f}%")

    # Train malicious model (copy of benign)
    malicious_model = SimpleCNN()
    malicious_model.load_state_dict(torch.load('models/benign_model.pth'))  # start from converged benign

    malicious_model = train_model(malicious_model, poisoned_train_loader, epochs=10)  # fewer epochs often enough

    # Save
    torch.save(malicious_model.state_dict(), 'models/malicious_model.pth')

    # Eval clean accuracy (should be similar to benign)
    malicious_clean_acc = evaluate(malicious_model, clean_test_loader)
    print(f"Malicious model clean accuracy: {malicious_clean_acc:.2f}%")

    # Eval ASR
    asr = asr_eval(malicious_model, poisoned_test_loader, target_class)
    print(f"Attack Success Rate (ASR): {asr:.2f}%")