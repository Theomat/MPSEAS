
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import glob


# =============================================================================
# Misc tunable parameters
# =============================================================================
marker_size = 110
axis_font_size = 15
legend_font_size = 15
# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Plot figures based on run data.")

argument_default_values = {
	"prefix": 'kissat_ibm',
}
argument_parser.add_argument('-p', '--prefix',
                             type=str,
                             action='store',
                             default=argument_default_values['prefix'],
                             help=f"File prefix or file path used in produce_run_data (default: '{argument_default_values['prefix']}')"
                             )
argument_parser.add_argument('-l', '--legend',
                             action='store_true',
                             dest='legend',
                             help=" (default: False)"
                             )
parsed_parameters = argument_parser.parse_args()

prefix: str = parsed_parameters.prefix.strip('/').split('/')[-1]
legend: bool = parsed_parameters.legend
# ==================================================================
# Try to load data file
# ==================================================================
file_prefix = "runs_" +prefix
print(f"Looking for data starting with prefix '{file_prefix}'...")
dataframes = []
for file in glob.glob("*.csv"):
    if file.startswith(file_prefix):
        # try:
            rest = file[len(file_prefix) + 1:-4].split("_")
            nb_configurations, ratio_instances = int(rest[0]), float(rest[1])
            df = pd.read_csv(file)
            df = df.drop("Unnamed: 0", axis=1)
            dataframes.append((nb_configurations, ratio_instances, df))
            print(f"\tfound for {nb_configurations} configurations and {ratio_instances*100}% of instances")
        # except:
            # pass
if len(dataframes) == 0:
    print("No dataframe was found!")
    import sys
    sys.exit(0)
# ==================================================================
# Set style
# ==================================================================
sns.set_style("whitegrid")
sns.color_palette("colorblind")
plt.style.use('seaborn-colorblind')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# ==================================================================
# FIELDS OF THE DATAFRAME
#     "time_used"
#     "total_time"
#     "current_challenger"
#     "errors"
#     "discarded"
#     "incumbent"
#     "seed"
#     "strategy"
# ==================================================================
def __preprocess__(df):
    df["selection"] = df["strategy"].str.extract(r'^([^+]*) \+ .*')
    df["discrimination"] = df["strategy"].str.extract(r'^[^+]* \+ (.*)')
    df["discrimination"] = df["discrimination"].str.replace(" 10100%", "")
    return df

dataframes = [(a, b, __preprocess__(df)) for (a,b ,df) in dataframes]
# ==================================================================
# Plot 
# ==================================================================

for (nb_conf, ratio, df) in dataframes:
    df["time_ratio"] = df["time_used"]  * 100/ df["total_time"]
    df: pd.DataFrame = df[["selection", "discrimination", "seed", "time_ratio", "discarded"]]
    df = df.groupby(["selection", "discrimination"]).mean().reset_index()

    markers = ["X", "d", "."][:len(np.unique(df["discrimination"]))]

    g = sns.relplot(x="time_ratio", y="discarded",
                    hue="selection", style="discrimination", data=df, s=marker_size, legend=legend,
                    markers=markers, linewidth=1)
    plt.xlabel("% of time", fontsize=axis_font_size)
    plt.ylabel("Number of challengers discarded", fontsize=axis_font_size)
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.show()

for (nb_conf, ratio, df) in dataframes:
    df["time_ratio"] = df["time_used"]  * 100/ df["total_time"]
    df: pd.DataFrame = df[["selection", "discrimination", "seed", "time_ratio", "errors"]]
    df = df.groupby(["selection", "discrimination"]).mean().reset_index()
    markers = ["X", "d", "."][:len(np.unique(df["discrimination"]))]

    g = sns.relplot(x="time_ratio", y="errors",
                    hue="selection", style="discrimination", data=df, s=marker_size, legend=legend,
                    markers=markers, linewidth=1)
    plt.xlabel("% of time", fontsize=axis_font_size)
    plt.ylabel("Number of prediction errors", fontsize=axis_font_size)
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.show()