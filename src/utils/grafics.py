from matplotlib import pyplot as plt


def graph_of_spector(w, i):
    plt.plot(w, i, label='Спектр', color='red')

    plt.xlabel('W')
    plt.ylabel('I')

    plt.show()