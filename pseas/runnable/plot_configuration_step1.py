"""
This file can generate all the figures used in the paper.
You can run it with -h for some help.

Feel free to change tunable parameters line 47 and after.
You can also do a bit of tweaking inside the methods.

Note that:
    - you need to have produced the data to be able to plot anything.
    - for the detailed_sata figures it may take a while.

"""
from typing import Optional, Tuple

from scipy.spatial import ConvexHull
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
parsed_parameters = argument_parser.parse_args()

folder: str = parsed_parameters.folder 
suffix: str = parsed_parameters.suffix
legend: bool = parsed_parameters.legend
# =============================================================================
# Finished parsing
# =============================================================================
# =============================================================================
# Start Tunable Parameters
# =============================================================================
marker_size = 110
axis_font_size = 15
legend_font_size = 15
# Data --------------------------------------------
# Paper Figures
general_perf: bool = True
general_perf_min_accuracy: float = 75

# Other Tables
bias: bool = True

# Detailed data ------------------------------------
# Paper Figures
accuracy_wrt_time: bool = True
correct_wrt_confidence: bool = True
instances_wrt_time: bool = True
# Other Figures
confidence_wrt_time: bool = True
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



def __rename_strategies__(df: pd.DataFrame) -> pd.DataFrame:
    df["strategy"] = df["strategy"].str.replace(
        ".*-discrimination-based", "discrimination-based", regex=True)
    df["strategy"] = df["strategy"].str.replace(
        "Info. over Decision/Time", "information-based", regex=False)
    df["strategy"] = df["strategy"].str.replace(
        "Random", "random", regex=False)

    # Rename discrimination component
    df["strategy"] = df["strategy"].str.replace(" 10100%", "", regex=False)
    df["strategy"] = df["strategy"].str.replace(".00%", "%", regex=False)
    df["strategy"] = df["strategy"].str.replace(
        "Subset", "subset", regex=False)

    df["selection"] = df["strategy"].str.extract(r'^([^+]*) \+ .*')
    df["discrimination"] = df["strategy"].str.extract(r'^[^+]* \+ (.*)')
    return df


def __filter_best_strategies__(df: pd.DataFrame) -> pd.DataFrame:
    # Remove all that don't have timeout correction
    df["baseline"] = df["selection"].str.contains(
        "random") | df["discrimination"].str.contains("subset")
    return df


def __name_filter__(df: pd.DataFrame, name_contains: str, apply_filter: Optional[bool]) -> pd.DataFrame:
    if apply_filter is not None:
        filter = df["strategy"].str.contains(name_contains)
        if not apply_filter:
            filter = ~filter
        df = df[filter]
    return df


def __filter_strategies_by_selection_method__(df: pd.DataFrame,
                                              baseline: Optional[bool] = None,
                                              discrimination_based: Optional[bool] = None,
                                              distribution_based: Optional[bool] = None,
                                              information_based: Optional[bool] = None,
                                              ) -> pd.DataFrame:
    df = __name_filter__(df, "random", baseline)
    df = __name_filter__(df, "discrimination-based", discrimination_based)
    df = __name_filter__(df, "variance-based", distribution_based)
    df = __name_filter__(df, "information-based", information_based)
    return df


def __filter_strategies_by_discriminator__(df: pd.DataFrame,
                                           baseline: Optional[bool] = None,
                                           wilcoxon: Optional[bool] = None,
                                           ) -> pd.DataFrame:
    df = __name_filter__(df, "subset", baseline)
    df = __name_filter__(df, "Wilcoxon", wilcoxon)
    return df


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


def plot_general_performance(df):
    # If data is missing for a dataset skip it
    if df.shape[0] == 0:
        return
    df["correct"] = df["y_pred"] == df["y_true"]

    # ==================================================
    #       CHANGE HERE FOR DIFFERENT STRATEGIES
    # ==================================================
    df = __filter_best_strategies__(df)
    # ==================================================
    # Take only interesting part
    df = df[["time", "correct", "selection", "discrimination"]]

    # Take mean over runs
    df = df.groupby(["selection", "discrimination"])
    df = df.agg({
        "time": "median",
        "correct": "mean"
    })
    print(df)
    df["time"] *= 100
    df["correct"] *= 100

    # Compute Pareto Front
    X = df["time"].to_numpy()
    Y = df["correct"].to_numpy()
    x, y = __pareto_front__(X, Y)
    print("Pareto front:")
    for px, py in zip(x, y):
        print(f"\tTime:{px:.4f}% Correct:{py:.2f}%")

    df = df.reset_index()
    markers = ["X", "d", "."][:np.unique(df["discrimination"].values).shape[0]]

    g = sns.relplot(x="time", y="correct", hue="selection", style="discrimination", data=df, s=marker_size, legend=legend,
                    markers=markers, linewidth=1)
    plt.plot(x, y, 'k--', label="Pareto front")
    plt.xlim(0, 100)
    plt.ylim(general_perf_min_accuracy, 101)
    plt.xlabel("% of time", fontsize=15)
    plt.ylabel("% of accuracy", fontsize=15)

    if legend:
        plt.setp(g.legend.get_texts(), fontsize=legend_font_size)
        plt.setp(g.legend.get_title(), fontsize=legend_font_size)
        figlegend = plt.figure(figsize=(2.51, 3.26))

        patches, labels = g.axes[0, 0].get_legend_handles_labels()
        # Get rid of the legend on the first plot, so it is only drawn on the separate figure
        g.legend.remove()
        labels[1] = r"$\bf{selection}$"
        labels[-3] = r"$\bf{discrimination}$"
        # # Add empty path with color white for better rendering
        # path = matplotlib.patches.PathPatch(
        #     matplotlib.path.Path(vertices=[(0, 0)]), facecolor="white")
        # patches.insert(7, path)
        # labels.insert(7, " ")

        figlegend.legend(handles=patches, labels=labels, ncol=1)
        figlegend.savefig('legend.pdf')
        print("Produced legend.pdf")
        plt.close(figlegend)
    plt.tight_layout()
    plt.show()




def print_table_bias(df: pd.DataFrame):
    # If data is missing for a dataset skip it
    if df.shape[0] == 0:
        return

    df["correct"] = df["y_pred"] == df["y_true"]
    # Compute direction of performance
    df["a_inc_is_better"] = df["perf_eval"] > df["perf_cmp"]
    df = df[["discrimination", "a_inc_is_better", "correct", "selection", "time"]]

    df["time"] *= 100
    df["correct"] *= 100
    df_aold = df[df["a_inc_is_better"] == True]
    df_anew = df[df["a_inc_is_better"] == False]
    out = df_aold.groupby(["selection", "discrimination"]).agg(
        {"time": "median", "correct": "mean"}).reset_index()
    out = out.rename(
        columns={"time": "time a_inc better", "correct": "correct a_inc better"})
    out = out.merge(df_anew.groupby(["selection", "discrimination"]).agg(
        {"time": "median", "correct": "mean"}).reset_index())
    out = out.rename(
        columns={"time": "time a_ch better", "correct": "correct a_ch better"})
    print(out)


def plot_correct_wrt_confidence(df: pd.DataFrame):
    # If data is missing for a dataset skip it
    if df.shape[0] == 0:
        return

    df = __filter_strategies_by_discriminator__(
        df, baseline=False)

    # Take mean performance
    df = df.groupby(["selection", "time"]).mean().reset_index()
    df["prediction"] *= 100

    g = sns.relplot(y="confidence", x="prediction",
                    hue="selection", data=df, legend=legend, marker=".")
    plt.plot(list(range(70, 101)), list(range(70, 101)), "black")
    plt.xlabel("% of accuracy", fontsize=axis_font_size)
    plt.ylabel("% of confidence", fontsize=axis_font_size)
    plt.xlim(70, 100)
    plt.ylim(70, 100)
    if legend:

        plt.setp(g.legend.get_texts(), fontsize=legend_font_size)
        plt.setp(g.legend.get_title(), fontsize=legend_font_size)
        figlegend = plt.figure(figsize=(2.61, 1.59))

        patches, labels = g.axes[0, 0].get_legend_handles_labels()
        # Get rid of the legend on the first plot, so it is only drawn on the separate figure
        g.legend.remove()

        figlegend.legend(handles=patches, labels=labels, ncol=1)
        figlegend.savefig('legend.pdf')
        print("Produced legend.pdf")
        plt.close(figlegend)
    plt.tight_layout()
    plt.show()


def plot_instances_wrt_time(df: pd.DataFrame):
    # If data is missing for a dataset skip it
    if df.shape[0] == 0:
        return

    df = __filter_strategies_by_discriminator__(
        df, baseline=False)
    # Remove Norm
    df = df[~df["selection"].str.contains("Norm")]
    # Take mean performance
    df = df.groupby(["selection", "time"]).mean().reset_index()

    g = sns.lineplot(x="time", y="instances",
                 hue="selection", data=df, legend=legend)
    plt.xlim(0, 100)
    plt.ylim(bottom=0, top=np.max(df["instances"].values))
    plt.xlabel("% of time", fontsize=axis_font_size)
    plt.ylabel("Number of instances run", fontsize=axis_font_size)
    if legend:
        plt.setp(g.get_legend().get_texts(), fontsize=legend_font_size)
        plt.setp(g.get_legend().get_title(), fontsize=legend_font_size)
        figlegend = plt.figure(figsize=(2.61, 1.59))

        patches, labels = g.get_legend_handles_labels()
        # Get rid of the legend on the first plot, so it is only drawn on the separate figure
        g.get_legend().remove()

        figlegend.legend(handles=patches, labels=labels, ncol=1)
        figlegend.savefig('legend.pdf')
        print("Produced legend.pdf")
        plt.close(figlegend)
    plt.tight_layout()
    plt.show()


def plot_confidence_wrt_time(df: pd.DataFrame):
    # If data is missing for a dataset skip it
    if df.shape[0] == 0:
        return

    df = __filter_strategies_by_discriminator__(
        df, baseline=False)

    # Take mean performance
    df = df.groupby(["selection", "time"]).mean().reset_index()

    g = sns.lineplot(x="time", y="confidence",
                 hue="selection", data=df, legend=legend)
    plt.xlabel("% of time", fontsize=axis_font_size)
    plt.ylabel("% of confidence", fontsize=axis_font_size)
    plt.xlim(0, 100)
    plt.ylim(75, 100)
    if legend:
        plt.setp(g.get_legend().get_texts(), fontsize=legend_font_size)
        plt.setp(g.get_legend().get_title(), fontsize=legend_font_size)
        figlegend = plt.figure(figsize=(2.61, 2.76))

        patches, labels = g.get_legend_handles_labels()
        # Get rid of the legend on the first plot, so it is only drawn on the separate figure
        g.get_legend().remove()
        figlegend.legend(handles=patches, labels=labels, ncol=1)
        figlegend.savefig('legend.pdf')
        print("Produced legend.pdf")
        plt.close(figlegend)
    plt.tight_layout()
    plt.show()


def plot_correct_wrt_time(df: pd.DataFrame):
    # If data is missing for a dataset skip it
    if df.shape[0] == 0:
        return

    df = __filter_best_strategies__(df)
    # Remove subset
    df = df[~df["discrimination"].str.contains("subset")]
    # Take mean performance
    df = df.groupby(["selection", "discrimination", "time"]
                    ).mean().reset_index()
    df["prediction"] *= 100

    g = sns.lineplot(x="time", y="prediction",
                 hue="selection", style="discrimination", data=df, legend=legend, linewidth=1)

    # plt.xlabel("% of instances", fontsize=axis_font_size)
    plt.xlabel("% of time", fontsize=axis_font_size)
    plt.ylabel("% of accuracy", fontsize=axis_font_size)
    plt.xlim(0, 100)
    plt.ylim(50, 100)
    plt.gca().set_aspect(2, 'box')


    if legend:
        plt.setp(g.get_legend().get_texts(), fontsize=legend_font_size)
        plt.setp(g.get_legend().get_title(), fontsize=legend_font_size)
        figlegend = plt.figure(figsize=(2.61, 2.76))

        patches, labels = g.get_legend_handles_labels()
        # Get rid of the legend on the first plot, so it is only drawn on the separate figure
        g.get_legend().remove()
        labels[0] = r"$\bf{selection}$"
        labels[-2] = r"$\bf{discrimination}$"
        # # Add empty path with color white for better rendering
        # path = matplotlib.patches.PathPatch(
        #     matplotlib.path.Path(vertices=[(0, 0)]), facecolor="white")
        # patches.insert(7, path)
        # labels.insert(7, " ")

        figlegend.legend(handles=patches, labels=labels, ncol=1)
        figlegend.savefig('legend.pdf')
        print("Produced legend.pdf")
        plt.close(figlegend)
    plt.tight_layout()
    plt.savefig("correct_wrt_time.pdf", bbox_inches='tight')
    print("Paper ready figure was saved as correct_wrt_time.pdf.")
    plt.show()


general_df = pd.read_csv(f"{folder}/runs_{suffix}.csv")
general_df = general_df.drop("Unnamed: 0", axis=1)
general_df = __rename_strategies__(general_df)
if general_perf:
    plot_general_performance(general_df)
if bias:
    print_table_bias(general_df)
del general_df  # Free memory
detailed_df = pd.read_csv(f"{folder}/detailed_runs_{suffix}.csv")
detailed_df = detailed_df.drop("Unnamed: 0", axis=1)
detailed_df = __rename_strategies__(detailed_df)
if correct_wrt_confidence:
    plot_correct_wrt_confidence(detailed_df)
if instances_wrt_time:
    plot_instances_wrt_time(detailed_df)
if confidence_wrt_time:
    plot_confidence_wrt_time(detailed_df)
if accuracy_wrt_time:
    plot_correct_wrt_time(detailed_df)
