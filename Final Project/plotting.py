from preprocess import *
import pandas
import matplotlib.pyplot as plt
import seaborn


def plot_for_linearity(path):
    data = pandas.read_csv(path)

    data = data[data['make'].str.contains("acura")]
    y = data['sellingprice']

    # Year-To-Price Scatter
    plt.scatter(data['year'], y)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('Year to Price Linearity Scatter')
    plt.savefig('images/year_to_price.png')
    plt.close()

    # Condition-To-Price Scatter
    plt.scatter(data['condition'], y)
    plt.xlabel('Condition')
    plt.ylabel('Price')
    plt.title('Condition to Price Linearity Scatter')
    plt.savefig('images/condition_to_price.png')
    plt.close()

    # Odometer-To-Price Scatter
    plt.scatter(data['odometer'], y)
    plt.xlabel('Odometer')
    plt.ylabel('Price')
    plt.title('Odometer to Price Linearity Scatter')
    plt.savefig('images/odometer_to_price.png')
    plt.close()

    # Mmr-To-Price Scatter
    plt.scatter(data['mmr'], y)
    plt.xlabel('MMR')
    plt.ylabel('Price')
    plt.title('MMR to Price Linearity Scatter')
    plt.savefig('images/mmr_to_price.png')
    plt.close()


def plot_heatmap(path):
    data = pandas.read_csv(path)
    corr = data.corr()
    seaborn.heatmap(corr, annot=True, cmap='RdYlGn')
    plt.title('Correlation Coefficient Heatmap')
    plt.savefig('images/heatmap.png')
    plt.close()


if __name__ == '__main__':
    plot_for_linearity(TRIMMED_DATA_PATH)
    plot_heatmap(TRIMMED_DATA_PATH)
