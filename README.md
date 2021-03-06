# MPSEAS

MPSEAS holds for Model-based Per Set Efficient Algorithm Selection and is a follow-up on the work that can be found in our [PSEAS repository](git@github.com:https://github.com/Theomat/PSEAS)

**Authors**:
Théo Matricon, [Marie Anastacio](https://www.universiteitleiden.nl/en/staffmembers/marie-anastacio#tab-1), [Holger Hoos](https://www.universiteitleiden.nl/en/staffmembers/holger-hoos#tab-1)

**Acknowledgement**:
[Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/), [Laurent Simon](https://www.labri.fr/perso/lsimon/)

## For details see our [CP 2021](https://cp2021.a4cp.org/#) paper:

[Statistical Comparison of Algorithm Performance Through Instance Selection](link):
![Figure](https://github.com/Theomat/PSEAS/raw/master/figure.png)

(TODO: update the following :)
## Usage

### Installation

```bash
# clone this repository

# create your new env
conda create -n pseas
# activate it
conda activate pseas
# install pip
yes | conda install pip
# install this package and the dependencies
pip install -e .
# you can also install dependencies with the following command where their versions are locked:
pip install -r requirements.txt

# You are good to go :)

# If you want to reproduce our experiments, you also need to clone the aslib data repository, which can be done this way:
git clone https://github.com/coseal/aslib_data.git
```

### File structure

```bash
pseas/
        data/                       # data loading and transformation
        discrimination/             # discrimination methods
        instance_selection/         # instance selection methods
        runnable/                   # experiments
        truncated_distributions/    # truncated distributions helper
```

### Reproducing the experiments

All of the files mentioned in this section are located in the folder ```pseas/runnable/```.
If a script has parameters then you can run the script with the option "-h" to get a short help, otherwise just running the script is enough.

To produce the run data that is necessary, you need to run for each dataset the ```produce_run_data.py``` file.

Then the resulting data can be visualised by running the ```plot_paper_figures.py``` file.

Here is a short summary of what each file does in ```pseas/runnable/```:

- ```extract_gluhack_data.py``` extracts the gluhack data from the SAT18 run data.
- ```plot_instances_algos_performances.py``` plots the ordered performance of all algorithms of the dataset, and also plots the ordered median time of each instance of the dataset.
- ```plot_paper_figures.py``` plots all of the figures included in the paper, and some that were not included if enabled.
- ```print_difficulty.py``` computes and print the difficulty of each dataset.
- ```print_distribution_likelihood.py``` computes and print the median log likelihood of each distribution for each dataset.
- ```produce_run_data.py``` produces the run data for one dataset.
- ```produce_topk_run_data.py``` produces the run data for the top k solvers for one dataset.
- ```strategy_comparator_helper.py``` just a helper class, running it does nothing.

### Quick guide to using strategies in your own environment

To create a ```Strategy``` you need the following files, that are independent of the rest:

```bash
pseas/discrimination/*
pseas/instance_selection/*
pseas/truncated_distributions/*
pseas/corrected_timeout_strategy.py
pseas/standard_strategy.py
pseas/strategy.py
```

Now you can create a strategy with:

```python
instance_selection = VarianceBased()    # Replace with your method
discrimination = Wilcoxon()             # Replace with your method
strategy = CorrectedTimeoutStrategy(instance_selection, discrimination, seed=42)
```

Then here is the following code to interact with your strategy:

```python
# Information is a dict containing all of the background information needed for the strategy.
# To see what is needed by your strategy take a look at the "ready" methods of your components.
# To see how we computed them see "pseas/data/prior_information.py" and the method "compute_all_prior_information".
strategy.ready(**information)
strategy.reset()

# The state is a tuple of two lists:
# - the first list contains for each instance the running time of the challenger.
# If the challenger has not been run on an instance then the running time here is None.
# - the second list contains the running times of the incumbent algorithm on each instance.
state = ([None for i in range(n_instances)], 
         [incumbent_running_time(i) for i in range(n_instances)])

# "feed" sends the information to the strategy and enable it to update its internal state.
strategy.feed(state)
# Or any other stopping criterion
while strategy.get_current_choice_confidence() < target_confidence:
    instance_to_run = strategy.choose_instance()
    # ... run challenger on instance_to_run and update state
    running_time_challenger = run_challenger(instance_to_run)
    state[0][instance_to_run] = running_time_challenger
    # ... end
    strategy.feed(state)
    print(f"Current confidence in choice:{strategy.get_current_choice_confidence() * 100):.2f}%")
    print("Believes challenger is better than incumbent:", strategy.is_better())
```

## Citing

TODO (waiting for publication)
