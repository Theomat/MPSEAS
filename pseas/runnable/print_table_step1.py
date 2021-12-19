import pandas as pd
import numpy as np
# =============================================================================
# Argument parsing.
# =============================================================================
import argparse

from scipy import integrate
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
parsed_parameters = argument_parser.parse_args()

folder: str = parsed_parameters.folder 
suffix: str = parsed_parameters.suffix
# =============================================================================
# Finished parsing
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


dico = {}
for i, configurations in enumerate(range(10, 60, 10)):
    for j, split in enumerate(range(10, 60, 10)):
        ratio = split / 100
        detailed_df = pd.read_csv(f"{folder}/detailed_runs_{suffix}_{configurations}_{ratio}.csv")
        detailed_df = detailed_df.drop("Unnamed: 0", axis=1)
        detailed_df = __rename_strategies__(detailed_df)
        df = __filter_best_strategies__(detailed_df)
        # Remove subset
        df = df[~df["discrimination"].str.contains("subset")]
        # Take mean performance
        df = df.groupby(["selection", "time"]).mean().reset_index()
        df["prediction"] *= 100

        for method in df["selection"].unique():
            if method not in dico:
                dico[method] = np.zeros((5, 5))

            data = df[df["selection"] == method]
            data = data[["prediction", "time"]].to_numpy()
            auc = integrate.trapezoid(data[:, 0], dx=1, axis=0)
            dico[method][i, j] = auc / 10000 * 100

for method, values in dico.items():
    print("\\begin{table}")
    print("\t\\centering")
    print("\t\\caption{Percentage of total AUC Evolution for " + method  + " on " + suffix.replace("_", " ") + "}")
    print("\t\\begin{tabular}{"+ ("c" * 6) + "}")
    print("\t\t\\toprule")
    print("\t\tConfigurations & 10 & 20 & 30 & 40 & 50 \\\\")
    for j, percent in enumerate(range(10, 60, 10)):
        print(f"\t\t{percent}\\% & " + " & ".join("{:.1f}".format(float(values[i, j])) for i, _ in enumerate(range(10, 60, 10))) + "\\\\")
    print("\t\t\\bottomrule")
    print("\t\\end{tabular}")
    print("\\end{table}")


