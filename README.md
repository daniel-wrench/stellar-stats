# `stellar_stats`: Analysing plasma observations from spacecraft while handling data gaps


## TO-DO (Daniel)
- ~~Simplify computation and plotting codes~~
- ~~Create tidy repo~~
- QUICK RE-EVALUATION OF IMMEDIATE PURPOSE OF WHAT I NEED
    - Sunpy doesn't work for velocity stuff, so will need to revert to cdflib code (see reynolds paper stuff)
- ~~Test out new numbered version on all 3 datasets, using parameters from my papers.

- Create the first 1 week of the Wind dataset
- Bring in codes from other repos and run the rest of the gapping pipeline for a small dataset
- Tidy up late-stage plotting scripts
- Consider integration of 
    - velocity (notes for Tulasi)
    - a new SF computation method, e.g. with gravity waves (notes for Alexa)
- (Run through Claude/Chat, first asking how best to get feedback - copilot in vs code might be most efficient?)
- Set up slurm scripts + instructions
- Create pyproject.toml file to replace requirements.txt and allow easier use of notebooks?

## How to run this code 

- You can find a guide to getting set up with Git, and slides from a workshop on reproducible research, in the `doc/` folder.
- See [here](https://github.com/DenisMot/Python-minimal-install) for a useful guide to installing a minimal Python environment and setting up in VS Code and conda.
- For more detail on Git, see [this great free tutorial](https://swcarpentry.github.io/git-novice/) 
- If you plan to develop a package, check out the [template repository for a Python package](https://github.com/UtrechtUniversity/re-python-package).

### You will need

-   Git (configured: check with `git config --list` )

    -   `git config --global user.name "Your Name"`

    -   `git config --global user.email "you@example.com"`

    -   For complete instructions, see `doc/git_setup_guide.md`

-   Python (latest stable version, e.g., 3.10+)

-   An active GitHub account

-   A terminal or IDE of your choice (VS Code recommended)

### Clone this repository 

```sh
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
```

### Change directory to this folder

```sh
cd YOUR-REPO-NAME
```

### Set up virtual environment
1. Create a virtual environment: `python -m venv venv`
2. Activate the environment:
    - Windows: `source venv/Scripts/activate`
    - Linux/macOS: `source venv/bin/activate` 
3. Install required packages: `pip install -r requirements.txt`
    - If you haven't yet set up this file, you can first install the required packages, then use `pip freeze > requirements.txt` to write all the dependencies (and their versions) to this file. *Note that this can be a bit overkill with the number of dependencies listed: see [here](https://calmcode.io/course/pip-tools/compile) for an improved method.*

### Set up data and variables
1. Download data using `notebooks\nasa_time_series_demo.ipynb`
1. Set spacecraft, variable names appropriately under "Entry point" near bottom of `01_COMPUTE_STATS.py`


### Execute code
*Note, you can also hit the **Play** button at the top of the script in VS code - they will automatically import the first file available*

Adjust `N_FILES` and `CONFIG_FILE` appropriately

```
for file_index in $(seq 0 N_FILES); do python -m scripts.run_compute configs.CONFIG_FILE $file_index; done 
```

# Plot a single interval and its statistics
```
python -m scripts.run_plot configs.CONFIG_FILE $file_index $interval_index 
```

## Project Structure
*Recommended: change based on your specific files. For example, you may also want a `src/`, `models/` or `notebooks/` folder. (If you are inside `notebooks/` you will likely want to put `sys.path.append(os.path.abspath(".."))` at the start of your notebook, so that you can import funcs from `scripts/`*

```

stellar-stats/         # Your project root
├── pipeline/            # THE LIBRARY (Core logic, importable everywhere)
│   ├── __init__.py      # Makes this folder a package
│   ├── compute.py       # High-level orchestration (run_pipeline)
│   ├── stats.py         # The math (ACF, SF, PSD, turbulence metrics)
│   ├── io.py            # CDF loading/saving, path management
│   └── utils.py         # Logging, decorators, helper functions
│
├── scripts/             # THE COMMAND CENTER (Production runs)
│   ├── run_compute.py   # Main CLI entry point
│   └── run_plot.py      
│
├── notebooks/           # THE LABORATORY (Exploration & Prototyping)
│   └── nasa_time_series_demo.ipynb     # Demo of downloading and plotting CDF data from NASA website
│
├── configs/             # THE SETTINGS (Experiment parameters)
│   ├── wind.py  
│   └── psp.py   
│
├── data/                # THE VAULT (Never tracked by Git)
│   ├── raw/             # Original CDFs (Read-only)
│   └── processed/       # Intermediate arrays/results
│
├── results/             # THE GALLERY (Outputs)
│   └── figures/            
│
├── .gitignore           # Excludes data/, results/, and __pycache__/
├── pyproject.toml       # Makes "pip install -e ." possible
├── LICENSE            <- License for open-source projects
│── CITATION.cff       <- Citation file for academic use
├── venv               <- Virtual environment (git ignored)
├── .gitignore         <- Files to ignore (e.g., __pycache__, logs, data)
└── README.md            # Project overview and setup instructions

```

## Share your software

1. Link Zenodo to your GitHub account
2. Use Zenodo to create a DOI for the current version of this repo
2. Use this DOI when creating a citation file for your repository using [cffinit](https://citation-file-format.github.io/cff-initializer-javascript/#/)

## License

This project is licensed under the terms of the [MIT License](/LICENSE).

## Notes
- Sunpy is good for easily downloading a bunch of files from a specific time period without worrying about file download commands etc.
- Its `TimeSeries()` function also makes it very convenient to read a CDF and convert it to a dataframe - HOWEVER it doesn't work for some files, e.g. Wind 3dp files (proton/electron moments). If you're just working with magnetic field data, then it's fine.