## Generative Adversarial networks (GAN)

The main idea in [GAN](https://arxiv.org/pdf/1406.2661.pdf) is to train generator (G) to generate synthetic data based on the input noise z  that closely resembles the real data.

This work serves as a starting point for implementing and training a  the GAN from scracth using linear layers.
It contain the different files below:
* requirements.txt : This file lists all the required dependencies and packages used in the project.
* dataset_loading.py :  This file is responsible for loading the dataset utilized in the project, specifically the MNIST dataset..
* model.py: This file contains the definitions of the two essential model classes, namely the Generator and the Discriminator. These classes define the architecture and functionality of the respective components of the GAN.
* train.py : This file encompasses all the necessary steps and processes involved in training the GAN model. It includes data loading, model instantiation, optimizer and loss function definition, training loop, and evaluation.
To be able to run the code as a script, you have  to (in your terminal):
* installl all the requirements
```bash
 pip install requirements.txt
```
* run the train.py file
```bash
$ python train.py
```