from preprocess import *
import pandas
import matplotlib.pyplot as plt


def plot_for_linearity(path):
    data = pandas.read_csv(path)

    # Trim data to be of only one car type (Allows for better check of whether data is linear or not)
    data = data[data['model'].str.contains('accord')]

    y = data['sellingprice']

    # Year-To-Price Scatter
    plt.scatter(data['year'], y)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('Year to Price Scatter')
    plt.savefig('images/year_to_price.png')
    plt.close()

    # Condition-To-Price Scatter
    plt.scatter(data['condition'], y)
    plt.xlabel('Condition')
    plt.ylabel('Price')
    plt.title('Condition to Price Scatter')
    plt.savefig('images/condition_to_price.png')
    plt.close()

    # Odometer-To-Price Scatter
    plt.scatter(data['odometer'], y)
    plt.xlabel('Odometer')
    plt.ylabel('Price')
    plt.title('Odometer to Price Scatter')
    plt.savefig('images/odometer_to_price.png')
    plt.close()

    # Mmr-To-Price Scatter
    plt.scatter(data['mmr'], y)
    plt.xlabel('MMR')
    plt.ylabel('Price')
    plt.title('MMR to Price Scatter')
    plt.savefig('images/mmr_to_price.png')
    plt.close()


if __name__ == '__main__':
    plot_for_linearity(TRIMMED_DATA_PATH)
