import pandas as pd


def read_data(dir:str):
    data = pd.read_csv(dir)
    return data


def process_columns(data:pd.DataFrame):
    # Convert the columns to lower case, remove symbols and then remove trailing whitespaces
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Remove non-alphanumeric characters
    data['store_id'] = data['store_id'].str.replace('\W', '')

    # Remove trailing underscores
    data.columns = data.columns.str.replace('_$', '')

    return data


def add_store_features(data:pd.DataFrame):
    # Group by store_id and calculate the sum for the specified columns
    store_sales = data.groupby('store_id')['trailing_24_month_sales'].sum().reset_index().rename(columns={'trailing_24_month_sales': 'store_trailing_24_month_sales'})
    store_units_sold = data.groupby('store_id')['trailing_24_month_units_sold'].sum().reset_index().rename(columns={'trailing_24_month_units_sold': 'store_trailing_24_month_units_sold'})
    store_inventory = data.groupby('store_id')['trailing_12_month_avg_inventory'].sum().reset_index().rename(columns={'trailing_12_month_avg_inventory': 'store_trailing_12_month_avg_inventory'})
    store_number_of_skus = data.groupby('store_id')['sku_id'].nunique().reset_index().rename(columns={'sku_id': 'store_number_of_skus'})

    # Merge the calculated sums back to the original dataframe
    data = data.merge(store_sales, on='store_id', how='left')
    data = data.merge(store_units_sold, on='store_id', how='left')
    data = data.merge(store_inventory, on='store_id', how='left')
    data = data.merge(store_number_of_skus, on='store_id', how='left')

    return data


def treatment_data_processing(data:pd.DataFrame, keep_columns:list=None):
    # Process the columns
    data = process_columns(data)

    # Add store features
    data = add_store_features(data)

    # Create presence matrix
    presence_matrix = data.pivot_table(index='store_id', columns='sku_id', aggfunc='size', fill_value=0)

    # Aggregate store features
    store_features = data[keep_columns].drop_duplicates()

    # Merge dataframes
    data = store_features.merge(presence_matrix, on='store_id', how='left').fillna(0)

    return data

def target_data_processing(data:pd.DataFrame, keep_columns:list=None):
    # Process the columns
    data = process_columns(data)

    # Add store features
    data = add_store_features(data)

    # Only keep the specified columns
    data = data[keep_columns]

    return data


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    dir = 'src/data/raw/store_sku_data.csv'
    treatment_columns = ['store_id',
                    'median_income', 'median_age',
                    'population', 'unemployment_rate',
                    'store_trailing_24_month_sales',
                    'store_trailing_24_month_units_sold',
                    'store_trailing_12_month_avg_inventory',
                    'store_number_of_skus']
    target_columns = ['store_id', 'sku_id',
                    'median_income', 'median_age',
                    'population', 'unemployment_rate',
                    'store_trailing_24_month_sales', 
                    'store_trailing_24_month_units_sold',
                    'store_trailing_12_month_avg_inventory',
                    'store_number_of_skus',
                    'trailing_6_month_gross_profit']
    data = read_data(dir)
    treatment_data = treatment_data_processing(data, treatment_columns)
    target_data = target_data_processing(data, target_columns)
    print(target_data.head())
