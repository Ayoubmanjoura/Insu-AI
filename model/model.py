import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Optimisation CPU avec Intel OneDNN
torch.set_num_interop_threads(torch.get_num_threads())

def load_data():
    data = pd.read_csv(r'.\model\data\training_data.csv')

    X = data.drop("Insulin_to_Inject", axis=1).values
    y = data["Insulin_to_Inject"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)

if __name__ == '__main__':
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler = load_data()

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)  # Fix Windows bug

    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)  # Fix Windows bug

    model = SimpleNN(X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = float("inf")
    patience = 10
    counter = 0

    epochs = 100
    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    test_loss = 0.0
    test_mae = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            test_mae += mae_criterion(outputs, labels).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mae = test_mae / len(test_loader)

    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"Test MAE: {avg_test_mae:.4f}")

    plt.plot(range(len(loss_history)), loss_history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler  # Save the scaler as well
    }, 'insu_ai_model.pth')

