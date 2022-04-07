import matplotlib.pyplot as plt
import pandas
import numpy as np

RAW_DATA_PATH = 'data/raw_data.csv'
TRIMMED_DATA_PATH = 'data/trimmed_data.csv'


def trim_raw_data(path):
    data = pandas.read_csv(path, skipinitialspace=True, on_bad_lines='skip')

    # Columns dropped because they are unique for each datapoint
    data.drop(columns=['vin', 'seller', 'saledate'], inplace=True)

    # Columns dropped because they are highly dependent on other features
    data.drop(columns=['trim'], inplace=True)

    # Unify Data to single formatting
    data['make'] = data['make'].str.lower()
    data['model'] = data['model'].str.lower()
    data['body'] = data['body'].str.lower()
    data['transmission'] = data['transmission'].str.lower()
    data['state'] = data['state'].str.lower()
    data['color'] = data['color'].str.strip("—")
    data['interior'] = data['interior'].str.strip("—")

    # Drop specific rows if they do not exist.
    data = data.dropna(subset=['year', 'make', 'body', 'condition', 'odometer', 'mmr', 'sellingprice'])

    # Remove outliers outside 3 standard deviations.
    data = data[np.abs(data.sellingprice - data.sellingprice.mean()) <= (3 * data.sellingprice.std())]
    data = data[np.abs(data.odometer - data.odometer.mean()) <= (3 * data.odometer.std())]
    data = data[np.abs(data.mmr - data.mmr.mean()) <= (3 * data.mmr.std())]

    # Histogram of no outliers
    plt.hist(data['sellingprice'], bins=75)
    plt.xlabel('Used Vehicle Sale Price')
    plt.ylabel('Frequency of Sale Price')
    plt.title('Sale Price Frequency After Preprocessing')
    plt.savefig('images/preprocessing.png')
    plt.close()

    data.to_csv(TRIMMED_DATA_PATH, index=False)


def generate_one_hot_data(path):
    # Set make, model, trim, body, transmission, state, color, and interior
    #   to be OneHotEncoded since they are categorized data
    data = pandas.read_csv(path)
    data = pandas.concat([data, pandas.get_dummies(data['make'])], axis=1).drop(columns=['make'])
    data = pandas.concat([data, pandas.get_dummies(data['model'])], axis=1).drop(columns=['model'])
    data = pandas.concat([data, pandas.get_dummies(data['body'])], axis=1).drop(columns=['body'])
    data = pandas.concat([data, pandas.get_dummies(data['transmission'])], axis=1).drop(columns=['transmission'])
    data = pandas.concat([data, pandas.get_dummies(data['state'])], axis=1).drop(columns=['state'])
    data = pandas.concat([data, pandas.get_dummies(data['color'])], axis=1).drop(columns=['color'])
    data = pandas.concat([data, pandas.get_dummies(data['interior'])], axis=1).drop(columns=['interior'])

    return data


if __name__ == "__main__":
    trim_raw_data(RAW_DATA_PATH)
