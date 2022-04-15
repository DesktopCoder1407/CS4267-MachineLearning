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


def plot_final_results():
    labels = ['Linear Regression', '\nGradient Boost', '\n\nRandom Forest', 'Light GBM', '\nXGBoost', '\n\nKMeans+LinReg',
              'Neural Network', '\nLinear Regression', 'SVM']
    colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue',
              'tab:blue', 'tab:orange', 'tab:orange']
    training_accuracy = [.87, .64, .98, .82, .81, .89, .85, .97, .97]
    test_accuracy = [.87, .64, .88, .81, .78, .88, .85, .97, .97]
    training_time_minutes = [15, 130, 75, 1.733, 180, 70, 600, .733, .633]

    # Plot covering Training Accuracy
    chart = plt.bar(labels, training_accuracy, color=colors)
    plt.bar_label(chart)

    plt.ylabel('R^2 Accuracy')
    plt.suptitle('Training Accuracies for All Algorithms')

    legend_colors = {'Baseline Algorithms': 'tab:blue', 'This Project\'s Algorithms': 'tab:orange'}
    legend_labels = list(legend_colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in legend_labels]
    plt.legend(handles, legend_labels, bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    plt.tight_layout()

    plt.savefig('images/results1.png')
    plt.close()

    # Plot covering Testing Accuracy
    chart = plt.bar(labels, test_accuracy, color=colors)
    plt.bar_label(chart)

    plt.ylabel('R^2 Accuracy')
    plt.suptitle('Test Accuracies for All Algorithms')

    legend_colors = {'Baseline Algorithms': 'tab:blue', 'This Project\'s Algorithms': 'tab:orange'}
    legend_labels = list(legend_colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in legend_labels]
    plt.legend(handles, legend_labels, bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    plt.tight_layout()

    plt.savefig('images/results2.png')
    plt.close()

    # Plot covering Training Time
    chart = plt.bar(labels, training_time_minutes, color=colors, log=True)
    plt.bar_label(chart)

    plt.ylabel('Training Time in Minutes')
    plt.suptitle('Training Time for All Algorithms in Minutes')

    legend_colors = {'Baseline Algorithms': 'tab:blue', 'This Project\'s Algorithms': 'tab:orange'}
    legend_labels = list(legend_colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in legend_labels]
    plt.legend(handles, legend_labels, bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    plt.tight_layout()

    plt.savefig('images/results3.png')
    plt.close()


if __name__ == '__main__':
    plot_for_linearity(TRIMMED_DATA_PATH)
    plot_heatmap(TRIMMED_DATA_PATH)
    plot_final_results()
