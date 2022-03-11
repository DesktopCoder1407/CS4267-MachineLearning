from preprocess import generate_one_hot_data
import matplotlib.pyplot as plt


def plot_for_linearity(path):
    data = generate_one_hot_data(path)
    y = data['Price']

    plt.scatter(data['Year'], y)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title('Year to Price Scatter')
    plt.savefig('images/year_to_price.png')

    plt.scatter(data['Mileage'], y)
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Mileage to Price Scatter')
    plt.savefig('images/mileage_to_price.png')


if __name__ == '__main__':
    plot_for_linearity('data/raw_listings_trimmed.csv')
