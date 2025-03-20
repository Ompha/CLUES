<p align="center">
  <a href="https://github.com/Ompha/CLUES/blob/main/logo.png">
    <img src="https://github.com/Ompha/CLUES/blob/main/logo.png" width="100%" />
  </a>
</p>

# CLUES
CLustering UnsupErvised with Sequencer: A fully interpretable clustering tool for analyzing spectral data 

[![DOI](https://zenodo.org/badge/820419517.svg)](https://zenodo.org/doi/10.5281/zenodo.13789620)

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
If you make use of the code in this directory, CLUES, please cite our paper, [Lu et al. 2025](https://ui.adsabs.harvard.edu/abs/2025ApJS..276...65L/abstract)  and the Sequencer paper ([Baron\& Menard 2020](https://ui.adsabs.harvard.edu/abs/2021ApJ...916...91B/abstract)).

Citation key:
```
@ARTICLE{2025ApJS..276...65L,
       author = {{Lu}, Cicero X. and {Mittal}, Tushar and {Chen}, Christine H. and {Li}, Alexis Y. and {Worthen}, Kadin and {Sargent}, B.~A. and {Lisse}, Carey M. and {Sloan}, G.~C. and {Hines}, Dean C. and {Watson}, Dan M. and {Rebollido}, Isabel and {Ren}, Bin B. and {Green}, Joel D.},
        title = "{Sequencing Silicates in the Spitzer Infrared Spectrograph Debris Disk Catalog. I. Methodology for Unsupervised Clustering}",
      journal = {\apjs},
     keywords = {Debris disks, Planetary system formation, Clustering, Silicate grains, Infrared spectroscopy, Spectroscopy, Exozodiacal dust, Minimum spanning tree, Planetesimals, Meteorites, Meteorite composition, 363, 1257, 1908, 1456, 2285, 1558, 500, 1950, 1259, 1038, 1037, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning},
         year = 2025,
        month = feb,
       volume = {276},
       number = {2},
          eid = {65},
        pages = {65},
          doi = {10.3847/1538-4365/ada0ba},
archivePrefix = {arXiv},
       eprint = {2501.01484},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJS..276...65L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```
```
@ARTICLE{2021ApJ...916...91B,
       author = {{Baron}, Dalya and {M{\'e}nard}, Brice},
        title = "{Extracting the Main Trend in a Data Set: The Sequencer Algorithm}",
      journal = {\apj},
     keywords = {Astronomy data analysis, 1858, Computer Science - Machine Learning, Astrophysics - Instrumentation and Methods for Astrophysics, Statistics - Machine Learning},
         year = 2021,
        month = aug,
       volume = {916},
       number = {2},
          eid = {91},
        pages = {91},
          doi = {10.3847/1538-4357/abfc4d},
archivePrefix = {arXiv},
       eprint = {2006.13948},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJ...916...91B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```


