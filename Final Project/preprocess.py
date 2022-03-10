import pandas

RAW_DATA_PATH = 'data/raw_listings.csv'
TRIMMED_DATA_PATH = 'data/raw_listings_trimmed.csv'
CLEAN_DATA_PATH = 'data/cleaned_listings.csv'


def trim_clean_raw(path):
    data = pandas.read_csv(path,  engine='python', skipinitialspace=True, on_bad_lines=trim_commas)
    data.drop(columns=['Id', 'Vin'], inplace=True)  # Columns dropped because they are unique to every datapoint.
    data.drop(columns=['City'], inplace=True)  # Column dropped because of its high uniqueness per datapoint.

    # Unify Data to single formatting
    data['State'] = data['State'].str.lower()
    data['Make'] = data['Make'].str.lower()
    data['Model'] = data['Model'].str.lower()

    data.to_csv(TRIMMED_DATA_PATH, index=False)


def trim_commas(bad_line: list[str] | None):
    return bad_line[:-1]


def generate_one_hot_data(path):
    # Set State, Make, and Model to be OneHotEncoded, since they are categorized data.
    data = pandas.read_csv(path)
    data = pandas.concat([data, pandas.get_dummies(data['State'])], axis=1).drop(columns=['State'])
    data = pandas.concat([data, pandas.get_dummies(data['Make'])], axis=1).drop(columns=['Make'])
    data = pandas.concat([data, pandas.get_dummies(data['Model'])], axis=1).drop(columns=['Model'])

    data.to_csv(CLEAN_DATA_PATH, index=False)


if __name__ == "__main__":
    # trim_clean_raw(RAW_DATA_PATH)
    generate_one_hot_data(TRIMMED_DATA_PATH)
