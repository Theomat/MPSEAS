"""
This file can generate all the figures used in the paper.
You can run it with -h for some help.

Note that:
    - you need to have produced the data to be able to plot anything.

"""
import numpy as np
import pandas as pd
# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Print median of additional time of step 2 based on run data.")

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
parsed_parameters = argument_parser.parse_args()

folder: str = parsed_parameters.folder 
suffix: str = parsed_parameters.suffix
# =============================================================================
# Finished parsing
# =============================================================================
METHODS = ["random","discrimination","variance","uncertainty", "udd"]


dico = {}
for i, configurations in enumerate(range(10, 60, 10)):
    for j, split in enumerate(range(10, 60, 10)):
        ratio = split / 100
        df = pd.read_csv(f"{folder}/selections_{suffix}_{configurations}_{ratio}.csv")
        df = df.drop("Unnamed: 0", axis=1)
        df = df.groupby(["selection", "seed"]).mean().reset_index()
        # Change here to MEAN or MEDIAN
        df = df.groupby(["selection"]).median().reset_index()
        df["rank"] = df["additional_time"].rank()
        print(df)
        for method in df["selection"].unique():
            if method not in dico:
                dico[method] = np.zeros((5, 5))

            data = df[df["selection"] == method]
            dico[method][i, j] = data["rank"].to_numpy()[0]


for method, values in dico.items():
    print("\\begin{table}")
    print("\t\\centering")
    print("\t\\caption{Rank of median for " + method  + " on " + suffix.replace("_", " ") + "}")
    print("\t\\begin{tabular}{"+ ("c" * 6) + "}")
    print("\t\t\\toprule")
    print("\t\tConfigurations & 10 & 20 & 30 & 40 & 50 \\\\")
    for j, percent in enumerate(range(10, 60, 10)):
        line_values = [float(values[i, j])
                       for i, _ in enumerate(range(10, 60, 10))]
        print(f"\t\t{percent}\\% & " + " & ".join(f"{val:.1f}" for val in line_values) + "\\\\")
    print("\t\t\\bottomrule")
    print("\t\\end{tabular}")
    print("\\end{table}")

print("\\begin{table}")
print("\t\\centering")
print("\t\\caption{Median Rank on " + suffix.replace("_", " ") + "}")
print("\t\\begin{tabular}{lr}")
print("\t\t\\toprule")
print("\t\tselection & rank \\\\")
for method, values in dico.items():
    print("\t\t"+method+" & "+str(np.median(values))+"\\\\")
print("\t\t\\bottomrule")
print("\t\\end{tabular}")
print("\\end{table}")
