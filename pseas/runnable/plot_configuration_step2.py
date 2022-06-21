"""
This file can generate all the figures used in the paper.
You can run it with -h for some help.

Feel free to change tunable parameters line 47 and after.
You can also do a bit of tweaking inside the methods.

Note that:
    - you need to have produced the data to be able to plot anything.
    - for the detailed_sata figures it may take a while.

"""
from typing import Tuple

from scipy.spatial import ConvexHull
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Plot figures based on run data.")

argument_default_values = {
	"suffix": 'kissat_ibm',
    "folder": "."
}

argument_parser.add_argument('-f', '--folder',
                             type=str,
                             action='store',
                             default=argument_default_values['folder'],
                             help="Ffolder in which to look for the file (default: '.')"
                             )
argument_parser.add_argument('-s', '--suffix',
                             type=str,
                             action='store',
                             default=argument_default_values['suffix'],
                             help="File suffix used in produce_run_data (default: 'kissat_ibm')"
                             )
argument_parser.add_argument('-l', '--legend',
                             action='store_true',
                             dest='legend',
                             help=" (default: False)"
                             )
argument_parser.add_argument('-n', '--no-show',
                             action='store_true',
                             dest='no_show',
                             help=" (default: False)"
                             )
parsed_parameters = argument_parser.parse_args()

folder: str = parsed_parameters.folder 
suffix: str = parsed_parameters.suffix
legend: bool = parsed_parameters.legend
no_show: bool = parsed_parameters.no_show
# =============================================================================
# Finished parsing
# =============================================================================
# =============================================================================
# Start Tunable Parameters
# =============================================================================
marker_size = 110
axis_font_size = 15
legend_font_size = 15
sns.set_style("whitegrid")
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# =============================================================================
# End Tunable Parameters
# =============================================================================

def __pareto_front__(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data: np.ndarray = np.zeros((X.shape[0], 2))
    data[:, 0] = X
    data[:, 1] = Y
    hull: ConvexHull = ConvexHull(data)
    points: np.ndarray = hull.points
    # Re order points to make a front
    x: np.ndarray = points[hull.vertices, 0]
    y: np.ndarray = points[hull.vertices, 1]
    indices = y.argsort()
    x = x[indices]
    y = y[indices]

    indice_x = x.argmin()

    front_x = [x[indice_x]]
    front_y = [y[indice_x]]

    x = np.delete(x, indice_x)
    y = np.delete(y, indice_x)

    index: int = 0
    while index < x.shape[0]:
        ax, ay = x[index], y[index]
        if index + 1 >= x.shape[0]:
            front_x.append(ax)
            front_y.append(ay)
            break
        bx, by = x[index + 1], y[index + 1]
        fx, fy = front_x[-1], front_y[-1]
        vax, vay = ax - fx, ay - fy
        vbx, vby = bx - fx, by - fy
        # Scalar product between normal (counter clockwise) and va
        if vby * vax - vbx * vay > 0:
            x[index] = bx
            y[index] = by
            index += 1
        else:
            index += 1
            if ay > front_y[-1]:
                front_x.append(ax)
                front_y.append(ay)

    return front_x, front_y




def plot_saved_time(df: pd.DataFrame):

    ax = sns.boxplot(x="selection", y="additional_time", data=df, width=0.8, order=["random","discrimination-1.12","variance","udd-1.4-0.2","uncertainty"], showfliers=False)
    ax.set_xticklabels(["random","discrimination","variance","udd","uncertainty"])
    ax.tick_params(labelsize="large", rotation=45)
    plt.ylabel("time used (s)",size='large')
    plt.semilogy()
    #plt.ylim([100,10000])
    if not no_show:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(suffix + "_selections.pdf")

general_df = pd.read_csv(f"{folder}/selections_{suffix}.csv")

general_df = general_df.groupby(["selection", "seed"]).mean().reset_index()

plot_saved_time(general_df)
