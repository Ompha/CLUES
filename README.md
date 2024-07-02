<p align="center">
  <a href="https://github.com/Ompha/CLUES/blob/main/logo.png">
    <img src="https://github.com/Ompha/CLUES/blob/main/logo.png" width="100%" />
  </a>
</p>

# CLUES
CLustering UnsupErvised with Sequencer: A fully interpretable clustering tool for analyzing spectral data 

## Welcome
Ha! You've stumbled upon this machine-learning classification tools for spectra and IFU data! 
## Quick Start
I recommend making a new conda environment for this software.
```
conda env create -n clues -f environment.yaml
conda activate clues
pip install TheSequencer
```
Follow the tutorial from notebooks folder. 
* To preprocess and prepare your data, see recommended practices in [data_Preprocessing.ipynb](https://github.com/Ompha/CLUES/blob/main/notebooks/data_Preprocessing.ipynb). 
* For analyzing and clustering your data, please refer to 
    * [ClusterAnalyses-1DSequence.ipynb](https://github.com/Ompha/CLUES/blob/main/notebooks/ClusterAnalyses-1DSequence.ipynb) where you can learn how to generate a 1D sequence using [Sequencer](https://github.com/dalya/Sequencer) (Baron & Menard 2020), and
    * [ClusterAnalyses-hierarchicalCluster.ipynb](https://github.com/Ompha/CLUES/blob/main/notebooks/ClusterAnalyses-hierarchicalCluster.ipynb) where you can perform hierarchical clustering and Sihouette Score criterion based on a Minimum spanning tree (MST). 

## Cite
If you make use of the code in this directory, CLUES, please cite our paper (in review) and the Sequencer paper.

Citation key:
```
@misc{baron2020extracting,
    title={Extracting the main trend in a dataset: the Sequencer algorithm},
    author={Dalya Baron and Brice Ménard},
    year={2020},
    eprint={2006.13948},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    year=2020
}
```


