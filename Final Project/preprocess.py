import pandas
import matplotlib.pyplot as plt

RAW_DATA_PATH = 'data/raw_data.csv'
TRIMMED_DATA_PATH = 'data/trimmed_data.csv'


def trim_raw_data(path):
    data = pandas.read_csv(path, skipinitialspace=True, on_bad_lines='skip')

    # Columns dropped because they are unique for each datapoint
    data.drop(columns=['vin', 'seller', 'saledate'], inplace=True)

    # Image of non-preprocessed data.
    data.hist(bins=100, layout=(2, 3))
    plt.suptitle('Before Processing')
    plt.savefig('images/before_preprocessing.png')

    # Unify Data to single formatting
    data['make'] = data['make'].str.lower()
    data['model'] = data['model'].str.lower()
    data['trim'] = data['trim'].str.lower()
    data['body'] = data['body'].str.lower()
    data['transmission'] = data['transmission'].str.lower()
    data['state'] = data['state'].str.lower()
    data['color'] = data['color'].str.lower()
    data['interior'] = data['interior'].str.lower()

    # Prune rows with missing data
    data.dropna(inplace=True)

    # Image of preprocessed data.
    data.hist(bins=100, layout=(2, 3))
    plt.suptitle('After Processing')
    plt.savefig('images/after_preprocessing.png')

    data.to_csv(TRIMMED_DATA_PATH, index=False)


def generate_one_hot_data(path):
    # Set make, model, trim, body, transmission, state, color, and interior
    #   to be OneHotEncoded since they are categorized data
    data = pandas.read_csv(path)
    data = pandas.concat([data, pandas.get_dummies(data['make'])], axis=1).drop(columns=['make'])
    data = pandas.concat([data, pandas.get_dummies(data['model'])], axis=1).drop(columns=['model'])
    data = pandas.concat([data, pandas.get_dummies(data['trim'])], axis=1).drop(columns=['trim'])
    data = pandas.concat([data, pandas.get_dummies(data['body'])], axis=1).drop(columns=['body'])
    data = pandas.concat([data, pandas.get_dummies(data['transmission'])], axis=1).drop(columns=['transmission'])
    data = pandas.concat([data, pandas.get_dummies(data['state'])], axis=1).drop(columns=['state'])
    data = pandas.concat([data, pandas.get_dummies(data['color'])], axis=1).drop(columns=['color'])
    data = pandas.concat([data, pandas.get_dummies(data['interior'])], axis=1).drop(columns=['interior'])

    return data


if __name__ == "__main__":
    trim_raw_data(RAW_DATA_PATH)
