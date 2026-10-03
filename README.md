# Topological Data Analysis and Machine Learning

Academic projects and tutorials connecting the shape of data with graph methods, persistent homology and machine learning.

The collection includes an academic group project on molecular classification, Mapper analyses of point clouds and speech features, and tutorial notebooks on topology and 3D shape classification.

## Start here

| File | Scope | Current reproduction status |
| --- | --- | --- |
| [Project 1.ipynb](Project%201.ipynb) | Group 4 project: molecular graph features, persistent-homology summaries, spectral features, Random Forest and SVM classification, including scaffold-based splitting. | Input CSV files are absent. Full rerun pending. |
| [feature_gen.py](feature_gen.py) | Molecular graph feature extraction used by Project 1. | Python module name corrected to match notebook imports. |
| [PROJECT_2.ipynb](PROJECT_2.ipynb) | Mapper implemented through overlapping intervals and DBSCAN; noisy annuli joined by a bridge. | Requires `data/noisy_annuli_bridge.csv`. |
| [project3 .ipynb](project3%20.ipynb) | Exploratory Mapper analysis of speech features using PCA, UMAP and t-SNE; node composition and graph summaries. | Requires `pd_speech_features.csv`. |
| [tutorial.ipynb](tutorial.ipynb) | Simplicial complexes, Rips complexes, Betti numbers and Betti curves. | Some later exercises require additional local datasets. |
| [classifying_shapes .ipynb](classifying_shapes%20.ipynb) | Persistent-homology features for supervised shape classification. | Helper modules are absent; saved outputs include execution errors. |
| [mapper_quickstart_new .ipynb](mapper_quickstart_new%20.ipynb) | Tutorial exploring giotto-tda's Mapper pipeline on generated data. | External dependency environment must be checked. |

## Methods

- Persistent homology, persistence entropy, amplitudes, Betti curves and Carlsson coordinates.
- Graph statistics and spectral features derived from graph Laplacians.
- Mapper with overlapping covers and density-based clustering.
- Supervised classification using Random Forest and SVM, with train/test splitting and evaluation metrics.

The molecular notebook includes scaffold-based partitioning to group structurally related molecules. Evaluation should preserve the distinction between this split and the other exploratory splits present in the notebook.

## Environment and inputs

`requirements.txt` lists dependencies identified in the notebooks. A complete compatible environment has not yet been validated for this collection.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

Run from the repository root so that the local `feature_gen` module is importable. Obtain the original project datasets before running the data-dependent notebooks. [data/README.md](data/README.md) lists the required files and their expected locations.

The feature generator currently substitutes zero-valued persistence features if giotto-tda is unavailable. Install the dependency and check the reported availability before interpreting an experiment as using topological features.

## Contributions and provenance

`Project 1.ipynb` identifies the molecular classification work as **Group 4, Project 1**. This repository is part of Thierry Kuate Mabap's academic portfolio and contains group work alongside training materials.

`mapper_quickstart_new .ipynb` identifies its source as the [giotto-tda Mapper example](https://github.com/giotto-ai/giotto-tda/blob/master/examples/mapper_quickstart.ipynb) and retains an **AGPLv3** notice. Tutorial references and attributions remain in the original notebooks.

## Next reproducibility milestones

Recover the exact datasets and shape-generation modules, validate a compatible environment, execute each project from a clean kernel, and record split definitions, seeds, metrics and limitations. No benchmark accuracy is asserted here from an unverified rerun.
