import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer
import numpy as np


from src.model.treatment_nn import TreatmentModel
from src.model.target_nn import TargetModel, prepare_bert_input
from src.data.data_pipeline import read_data, treatment_data_processing, target_data_processing

class CausalInference():
    def __init__(self, data_dir:str, treatment_columns:list, target_columns:list):
        self.data_dir = data_dir
        self.treatment_columns = treatment_columns
        self.target_columns = target_columns

    def get_treatment_data(self):
        treatment_columns = ['store_id',
                             'median_income', 'median_age',
                             'population', 'unemployment_rate',
                             'store_trailing_24_month_sales',
                             'store_trailing_24_month_units_sold',
                             'store_trailing_12_month_avg_inventory',
                             'store_number_of_skus']
        data = read_data(self.data_dir)
        final_data = treatment_data_processing(data, treatment_columns)
        #find list of all columns that have 'SKU' in them
        sku_columns = [col for col in final_data.columns if 'SKU' in col]

        return final_data, sku_columns

    def process_treatment_data(self):
        final_data, sku_columns = self.get_treatment_data()

        # Normalize the input features
        self.treatment_scaler = StandardScaler()
        X = self.treatment_scaler.fit_transform(final_data.drop(columns=['store_id']).values)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(final_data[sku_columns].values, dtype=torch.float32)

        return X, y
    
    def get_target_data(self):
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
        data = read_data(self.data_dir)
        final_data = target_data_processing(data, target_columns)
        return final_data
    
    def process_target_data(self):
        final_data = self.get_target_data()
    
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
        self.target_scaler = StandardScaler()
        scaled_numeric_features = self.target_scaler.fit_transform(final_data[numeric_features])

        # Combine non-text features
        non_text_features = np.hstack([scaled_numeric_features, encoded_categorical_features])

        # Convert to tensor
        X_non_text = torch.tensor(non_text_features, dtype=torch.float32)
        y = torch.tensor(final_data[target_column].values, dtype=torch.float32).unsqueeze(1)  # Ensure y is 2D with shape (n_samples, 1)

        return X_non_text, input_ids, attention_mask, y
    
    def train_treatment_model(self):
        X, y = self.process_treatment_data()
        treatment_model = TreatmentModel(input_dim=X.shape[1], output_dim=y.shape[1])
        treatment_model.train(X, y)
        return treatment_model
    
    def train_target_model(self):
        X_non_text, input_ids, attention_mask, y = self.process_target_data()
        target_model = TargetModel(input_dim=X_non_text.shape[1], bert_model_name='bert-base-uncased')
        target_model.train(X_non_text, input_ids, attention_mask, y)
        return target_model
    
    def assemble_target_inferential_data(self):
        final_data = self.get_target_data()
        action_columns = ['sku_id', 'sku_description', 'sku_color', 'sku_material']
        store_columns = ['store_id','median_income', 'median_age',
                         'population', 'unemployment_rate',
                         'store_trailing_24_month_sales',
                         'store_trailing_24_month_units_sold',
                         'store_trailing_12_month_avg_inventory',
                         'store_number_of_skus']
        
        #get distinct rows for action columns
        action_df = final_data[action_columns].drop_duplicates().reset_index(drop=True)
        #get distinct rows for store columns
        store_df = final_data[store_columns].drop_duplicates().reset_index(drop=True)

        #get all possible combinations of action and store columns
        action_df['key'] = 1
        store_df['key'] = 1
        final_data = pd.merge(action_df, store_df, on='key').drop(columns='key')

        return final_data
    
    def infer_target_model(self):
        target_model = self.train_target_model()
        final_data = self.assemble_target_inferential_data()
        # Separate features
        categorical_features = ['sku_color', 'sku_material']
        text_features = ['sku_description']
        numeric_features = final_data.drop(columns=categorical_features + text_features + ['store_id', 'sku_id']).columns.tolist()

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
        scaled_numeric_features = self.target_scaler.transform(final_data[numeric_features])

        # Combine non-text features
        non_text_features = np.hstack([scaled_numeric_features, encoded_categorical_features])

        # Convert to tensor
        X_non_text = torch.tensor(non_text_features, dtype=torch.float32)

        predictions = target_model.infer(X_non_text, input_ids, attention_mask)

        #concatenate the predictions with the final_data
        final_data['predicted_gross_profit'] = predictions.numpy()

        return final_data

        
    
