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
    def __init__(self, data_dir:str):
        self.data_dir = data_dir

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
        bert_model_name = 'bert-base-uncased'
        X_non_text, input_ids, attention_mask, y = self.process_target_data()
        target_model = TargetModel(non_text_input_dim=X_non_text.shape[1], bert_model_name=bert_model_name, output_dim=1)
        target_model.train(X_non_text, input_ids, attention_mask, y, num_epochs=100, batch_size=15)
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
    
    def infer_treatment_model(self):
        treatment_model = self.train_treatment_model()
        final_data, sku_columns = self.get_treatment_data()
        # Normalize the input features
        X = self.treatment_scaler.transform(final_data.drop(columns=['store_id']).values)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(final_data[sku_columns].values, dtype=torch.float32)

        predictions = treatment_model.infer(X)

        #concatenate the predictions with the final_data
        final_data[sku_columns] = predictions.numpy()

        #keep only the store_id and the predictions
        final_data = final_data[['store_id'] + sku_columns]

        #pivot the data wide to long
        final_data = final_data.melt(id_vars='store_id', var_name='sku_id', value_name='sku_confidence_score')

        return final_data
    
    def optim_and_control_data(self, confidence_threshold=0.0):
        treatment_data = self.infer_treatment_model()
        target_data = self.infer_target_model()

        #merge the treatment and target data
        data = treatment_data.merge(target_data, on=['store_id', 'sku_id'], how='inner')

        #order by store_id and predicted_gross_profit descending
        optim_data = data.sort_values(['store_id', 'predicted_gross_profit'], ascending=[True, False])

        #only keep the rows where sku_confidence_score is greater than or equal to confidence_threshold
        optim_data = optim_data[optim_data['sku_confidence_score'] >= confidence_threshold].reset_index(drop=True)

        #group by store_id and add a column for the rank
        optim_data['rank'] = optim_data.groupby('store_id').cumcount() + 1

        #only keep the rows where rank is less than or equal to 'store_number_of_skus'
        optim_data = optim_data[optim_data['rank'] <= optim_data['store_number_of_skus']]

        #remove the rank column
        optim_data = optim_data.drop(columns='rank')

        #create the control data
        #get the target data
        control_data = self.get_target_data()
        #only keep the store_id and sku_id columns
        control_data = control_data[['store_id', 'sku_id']]
        #merge the control data with the data
        control_data = control_data.merge(data, on=['store_id', 'sku_id'], how='inner')

        #only keep the columns we need
        control_data = control_data[['store_id', 'sku_id', 'sku_confidence_score', 'predicted_gross_profit']]
        optim_data = optim_data[['store_id', 'sku_id', 'sku_confidence_score', 'predicted_gross_profit']]

        return control_data, optim_data
    
    def evaluate(self, control_data, optim_data):
        #only keep columns we need
        control_data = control_data[['store_id', 'predicted_gross_profit']]
        optim_data = optim_data[['store_id', 'predicted_gross_profit']]
        #group by store_id and sum the predicted_gross_profit
        control_total_gross_profit = control_data.groupby('store_id')['predicted_gross_profit'].sum().reset_index()
        optim_total_gross_profit = optim_data.groupby('store_id')['predicted_gross_profit'].sum().reset_index()

        #change the name of the columns predicted_gross_profit to control_gross_profit and optim_gross_profit
        control_total_gross_profit.columns = ['store_id', 'control_gross_profit']
        optim_total_gross_profit.columns = ['store_id', 'optim_gross_profit']

        #merge the data on store_id
        merged_data = control_total_gross_profit.merge(optim_total_gross_profit, on='store_id', how='inner')

        #add percentage increase column
        merged_data['percentage_increase'] = ((merged_data['optim_gross_profit'] - merged_data['control_gross_profit']) / merged_data['control_gross_profit']) * 100

        return merged_data
    

if __name__ == '__main__':
    causal_inference = CausalInference('src/data/raw/store_sku_data.csv')
    control_data, optim_data = causal_inference.optim_and_control_data()
    merged_data = causal_inference.evaluate(control_data, optim_data)
    print(merged_data)
    #save the control and optim data
    control_data.to_csv('src/data/outputs/control_data.csv', index=False)
    optim_data.to_csv('src/data/outputs/optim_data.csv', index=False)
    merged_data.to_csv('src/data/outputs/merged_data.csv', index=False)

        
    
