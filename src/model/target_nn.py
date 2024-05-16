import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from transformers import BertModel, BertTokenizer

# Define the neural network for non-text features
class NonTextNN(nn.Module):
    def __init__(self, input_dim):
        super(NonTextNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return x

# Define the neural network for text features (using BERT)
class TextNN(nn.Module):
    def __init__(self, bert_model_name):
        super(TextNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get the pooled output (CLS token)
        x = torch.relu(self.fc1(pooled_output))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return x

# Define the combined neural network
class CombinedNN(nn.Module):
    def __init__(self, non_text_input_dim, bert_model_name, output_dim):
        super(CombinedNN, self).__init__()
        self.non_text_nn = NonTextNN(non_text_input_dim)
        self.text_nn = TextNN(bert_model_name)
        self.fc1 = nn.Linear(64 + 64, 64)  # Combining the outputs of the two networks
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, non_text_x, input_ids, attention_mask):
        non_text_output = self.non_text_nn(non_text_x)
        text_output = self.text_nn(input_ids, attention_mask)
        combined = torch.cat((non_text_output, text_output), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        x = self.dropout(x)
        return x  # No activation function for regression output


class TargetModel:
    def __init__(self, non_text_input_dim, bert_model_name, output_dim, lr=0.001, patience=50):
        self.model = CombinedNN(non_text_input_dim=non_text_input_dim, bert_model_name=bert_model_name, output_dim=output_dim)
        self.criterion = nn.MSELoss()  # Use MSELoss for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.patience = patience
        self.model_trained = False
        # Generate a random id for the model
        self.model_id = np.random.randint(0, 1000000)

    def train(self, X_train_non_text, input_ids_train, attention_mask_train, y_train, X_val_non_text=None, input_ids_val=None, attention_mask_val=None, y_val=None, num_epochs=100, batch_size=10):
        train_dataset = torch.utils.data.TensorDataset(X_train_non_text, input_ids_train, attention_mask_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        if X_val_non_text is not None and input_ids_val is not None and attention_mask_val is not None and y_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val_non_text, input_ids_val, attention_mask_val, y_val)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        best_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_non_text_x, batch_input_ids, batch_attention_mask, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_non_text_x, batch_input_ids, batch_attention_mask)
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
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), f'src/model/serialized_models/target_model_{self.model_id}.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print("Early stopping!")
                        break
            
            else:
                loss = epoch_loss / len(train_loader)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.model.state_dict(), 'src/model/serialized_models/best_target_model.pth')
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
            for batch_non_text_x, batch_input_ids, batch_attention_mask, batch_y in data_loader:
                outputs = self.model(batch_non_text_x, batch_input_ids, batch_attention_mask)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        return val_loss / len(data_loader)

    def infer(self, X_non_text, input_ids, attention_mask):
        if not self.model_trained:
            raise Exception("Model is not trained yet. Please train the model before inference.")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_non_text, input_ids, attention_mask)
        return predictions
    
    def residual(self, X_non_text, input_ids, attention_mask, y):
        if not self.model_trained:
            raise Exception("Model is not trained yet. Please train the model before inference.")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_non_text, input_ids, attention_mask)
            residuals = y - predictions
        return residuals

    def load_model(self, path='src/model/serialized_models/best_target_model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model_trained = True

def prepare_bert_input(texts, tokenizer, max_length=128):
    inputs = tokenizer(texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    return inputs['input_ids'], inputs['attention_mask']

if __name__ == '__main__':
    from src.data.data_pipeline import read_data, target_data_processing
    dir = 'src/data/raw/store_sku_data.csv'
    target_columns = ['store_id', 'sku_id',
                    'sku_color', 'sku_material',
                    'sku_description', 'median_income',
                    'median_age', 'population',
                    'unemployment_rate',
                    'store_trailing_24_month_sales', 
                    'store_trailing_24_month_units_sold',
                    'store_trailing_12_month_avg_inventory',
                    'store_number_of_skus',
                    'trailing_6_month_gross_profit']
    data = read_data(dir)
    final_data = target_data_processing(data, target_columns)
    
    # Define the target column
    target_column = 'trailing_6_month_gross_profit'
    
    # Separate features
    categorical_features = ['sku_color', 'sku_material']
    text_features = ['sku_description']
    numeric_features = final_data.drop(columns=categorical_features + text_features + ['store_id', 'sku_id', target_column]).columns.tolist()

    # One-Hot Encoding for categorical features
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    encoded_categorical_features = encoder.fit_transform(final_data[categorical_features])

    # Ensure encoded categorical features are 2D
    if len(encoded_categorical_features.shape) == 1:
        encoded_categorical_features = encoded_categorical_features.reshape(-1, 1)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model_name = 'bert-base-uncased'

    # Prepare BERT input
    input_ids, attention_mask = prepare_bert_input(final_data['sku_description'], tokenizer)

    # Standardize numeric features
    scaler = StandardScaler()
    scaled_numeric_features = scaler.fit_transform(final_data[numeric_features])

    # Combine non-text features
    non_text_features = np.hstack([scaled_numeric_features, encoded_categorical_features])

    # Convert to tensor
    X_non_text = torch.tensor(non_text_features, dtype=torch.float32)
    y = torch.tensor(final_data[target_column].values, dtype=torch.float32).unsqueeze(1)  # Ensure y is 2D with shape (n_samples, 1)

    # Initialize and train the model
    target_model = TargetModel(non_text_input_dim=X_non_text.shape[1], bert_model_name=bert_model_name, output_dim=1)
    target_model.train(X_non_text, input_ids, attention_mask, y, num_epochs=100, batch_size=15)

    # Load the best model
    target_model.load_model()

    # Perform inference
    predictions = target_model.infer(X_non_text, input_ids, attention_mask)
    print(predictions)

    # Calculate residuals
    residuals = target_model.residual(X_non_text, input_ids, attention_mask, y)
    print(residuals)


