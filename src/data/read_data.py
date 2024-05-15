import pandas as pd


def read_data(dir:str):
    data = pd.read_csv(dir)
    return data


def process_data_columns(data:pd.DataFrame):
    # Convert the columns to lower case, remove symbols and then remove trailing whitespaces
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Remove non-alphanumeric characters
    data['store_id'] = data['store_id'].str.replace('\W', '')

    # Remove trailing underscores
    data.columns = data.columns.str.replace('_$', '')

    return data


def add_store_features(data:pd.DataFrame):
    # Group by store_id and calculate the sum for the specified columns
    store_sales = data.groupby('store_id')['trailing_24_month_sales'].sum().reset_index().rename(columns={'trailing_24_month_sales_': 'store_trailing_24_month_sales'})
    store_units_sold = data.groupby('store_id')['trailing_24_month_units_sold'].sum().reset_index().rename(columns={'trailing_24_month_units_sold': 'store_trailing_24_month_units_sold'})

    # Merge the calculated sums back to the original dataframe
    data = data.merge(store_sales, on='store_id', how='left')
    data = data.merge(store_units_sold, on='store_id', how='left')

    return data


def data_preprocessing(data:pd.DataFrame):
    # Process the columns
    data = process_data_columns(data)

    # Add store features
    data = add_store_features(data)

    return data


if __name__ == '__main__':
    dir = 'src/data/raw/store_sku_data.csv'
    data = read_data(dir)
    data = data_preprocessing(data)
    print(data.head())
