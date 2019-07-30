# Going deep into Human Activity Recognition

![py](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python)
![jn](https://img.shields.io/badge/Jupyter-enabled-orange.svg?logo=Jupyter)
![l1](https://img.shields.io/badge/License-GPL%203.0-brightgreen.svg)
![l2](https://img.shields.io/badge/License-CC%20BY--SA%204.0-red.svg)

Human Activity Recognition (HAR) has become a key research topic in monitored and assisted living either for medical or tracking reasons.

First attempts provided manual feature crafting, followed by analysis done either with deep neural networks or other approaches like Hidden Markov models.
More recently instead, direct analysis on raw signals has been attempted.
Here we continue this trend by exploring some possible approaches with convolutional and recurrent neural networks and look over automatic feature extraction techniques, such as autoencoders.
Most of the datasets in this fields are highly unbalanced and some classes lack of enough data.
To face this, we propose two augmentation techniques for rebalancing.
Finally, we introduce new ways and metrics to select the best learning epoch to address overfitting and get the best learning results overall.

Our tests confirm that augmenting the initial dataset is worth the effort, and we achieve performance that surpass what is declared for it.
Moreover, we discovered that working with raw signals in the sensor reference frame is better than working with their trasformation to the body frame.
As for encoded data by the means of our autoencoders, we could not find any performance improvement: in some cases, worse results are obtained.

This is the final project for the course of Human Data Analytics, Spring 2019, Master's degree in ICT for Internet and multimedia, University of Padova, Italy.

## Technical details

Download the dataset into the `dataset` directory (follow the [readme](dataset/README.md)).

Open the Jupyter notebooks to start the journey, from data preprocessing to the final results.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliabntt/REPO/blob/master/HAR-1-Preprocessing.ipynb) Part 1 - Preprocessing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliabntt/REPO/blob/master/HAR-2-Models.ipynb) Part 2 - Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliabntt/REPO/blob/master/HAR-3-SVM.ipynb) Part 3 - SVM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliabntt/REPO/blob/master/HAR-4-Visualization.ipynb) Part 4 - Visualization

More complete information can be found in the [full report](report/report.pdf).

**Important:** Colab notebooks opened from Github do not allow for mounting other directories, the dataset must be uploaded to Drive.

## Licenses

Copyright (C) 2019 Elia Bonetto and Filippo Rigotto.

Code released under [GPL v3.0](LICENSE), report under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en).
