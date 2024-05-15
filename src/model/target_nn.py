import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Define a simple feedforward neural network
class RegressionNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x  # No activation function for regression output

class TargetModel:
    def __init__(self, input_dim, output_dim, lr=0.001, patience=10):
        self.model = RegressionNN(input_dim=input_dim, output_dim=output_dim)
        self.criterion = nn.MSELoss()  # Use MSELoss for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.patience = patience
        self.model_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None, num_epochs=100, batch_size=2):
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print("Early stopping!")
                        break

        self.model_trained = True

    def _evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        return val_loss / len(data_loader)

    def infer(self, X):
        if not self.model_trained:
            raise Exception("Model is not trained yet. Please train the model before inference.")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions
    
    def residual(self, X, y):
        if not self.model_trained:
            raise Exception("Model is not trained yet. Please train the model before inference.")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            residuals = y - predictions
        return residuals

    def load_model(self, path='best_model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model_trained = True


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from src.data.data_pipeline import read_data, target_data_processing
    dir = 'src/data/raw/store_sku_data.csv'
    target_columns = ['store_id', 'sku_id',
                    'median_income', 'median_age',
                    'population', 'unemployment_rate',
                    'store_trailing_24_month_sales', 
                    'store_trailing_24_month_units_sold',
                    'store_trailing_12_month_avg_inventory',
                    'store_number_of_skus',
                    'trailing_6_month_gross_profit']
    data = read_data(dir)
    final_data = target_data_processing(data, target_columns)
    
    # Define the target column
    target_column = 'trailing_6_month_gross_profit'

    # Normalize the input features
    scaler = StandardScaler()
    X = scaler.fit_transform(final_data.drop(columns=['store_id', 'sku_id', target_column]).values)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(final_data[target_column].values, dtype=torch.float32).unsqueeze(1)  # Ensure y is 2D with shape (n_samples, 1)

    # Initialize and train the model
    target_model = TargetModel(input_dim=X.shape[1], output_dim=1)
    target_model.train(X, y)

    residual = target_model.residual(X,y)
    print(residual)