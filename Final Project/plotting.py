from preprocess import *
import pandas
import matplotlib.pyplot as plt
import seaborn


def plot_for_linearity(path):
    data = pandas.read_csv(path)

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

    #Condensed Linearity Scatter
    x_vars = ['year', 'condition', 'odometer', 'mmr']
    y_vars = ['sellingprice']
    g = seaborn.PairGrid(pandas.read_csv(TRIMMED_DATA_PATH), x_vars=x_vars, y_vars=y_vars, height=3)
    g.map(seaborn.scatterplot)
    plt.savefig('images/linearity.png')
    plt.close()


def plot_heatmap(path):
    data = pandas.read_csv(path)
    corr = data.corr()
    seaborn.heatmap(corr, annot=True, cmap='RdYlGn')
    plt.savefig('images/heatmap.png')
    plt.close()


if __name__ == '__main__':
    plot_for_linearity(TRIMMED_DATA_PATH)
    plot_heatmap(TRIMMED_DATA_PATH)
