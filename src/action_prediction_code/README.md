Copyright [2016] [Fang Wang]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


# Introduction

This package includes the work of "Prediction of Manipulation Action", please cite the reference below if you use this code for your work.

@article{action_prediction_2016,
  title={Prediction of Manipulation Actions},
  author={Cornelia Fermuller, Fang Wang, Yezhou Yang, Konstantinos Zampogiannis, Yi Zhang, Francisco Barranco and Michael Pfeiffer},
  eprint = {arXiv:1608.xxxx},
  year={2016},
}

# Usage

Please download the data and labels from our [project page](http://users.cecs.anu.edu.au/~fwang/action_prediction/) and extract the data files in the ''dataset'' folder. The directory organization should be similar as shown below:

    dataset
     |---- MAD
     |      |---- cup_feat.mat
     |      |---- cup_db.json
     |      |---- ...
     |
     |---- HAF
            |---- fbr_feats.mat
            |---- fbr_db.json
            |---- ...

We tested the code on Python 2.7 with Theano 0.8.2. Please refer to the official site for further information of [Theano](http://deeplearning.net/software/theano/). We also have some Matlab scripts for evaluation. The codes have been tested only on MATLAB R2014b, but it should be working on older version.


1. Train the action prediction model
    python LSTMAction.py

2. Train the force regression model
    python LSTMRegressor.py

3. Evaluation tools
    We provided two matlab scripts for evaluation.
    - action_eval.m  This script calculates the prediction accuracy of trained model and draws the confusion matrix.
    - plot_force_samples.m  For force estimation, we already included the evaluation code in the python script. This tool is used for visualize the force regression results for random selected samples.


# Reference

Please refer to our project site for more details.
[http://users.cecs.anu.edu.au/~fwang/action_prediction/](http://users.cecs.anu.edu.au/~fwang/action_prediction/)

If you find any problem or questions, please email [Fang Wang] [fang.wang@anu.edu.au].

