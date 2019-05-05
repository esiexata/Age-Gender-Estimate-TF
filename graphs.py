import matplotlib.pyplot as plt


def graphbar(rangesF,rangesM):


    # Credit: Josh Hemann

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple


    n_groups = 5

    means_men = (rangesM)
    std_men = (0, 0, 0, 0, 0)

    means_women = (rangesF)
    std_women = (0, 0, 0, 0, 0)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means_men, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_men, error_kw=error_config,
                    label='Men')

    rects2 = ax.bar(index + bar_width, means_women, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_women, error_kw=error_config,
                    label='Women')

    ax.set_xlabel('Group')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
    ax.legend()

    fig.tight_layout()
    plt.show()