# Distributed Learning in Multi-Agent Systems
## Introduction
This code compares the accuracy or full network topology with the ring network topology in decentralised learing systems. The study uses the MNIST dataset which is divided into IID and non-IID subsets. 
This code also has two impplementations of the consensus aolgorithm.

## Acknowledgement and referencing
Some code snippets were taken from https://medium.com/data-science/math-neural-network-from-scratch-in-python-d6da9f29ce65 with some modification.


## Setup
 1 First install the code as a zip and place unpack.
 
 2 Extract the MNIST folder to a known path as it includes all the MNIST image data.
 
 3 Update the File Paths in LoadData.py to the appropriate file paths
 
    Lines 26 to 29 and Lines 90 to 93

 4 Install Necessary Libraries
 
    Those Include Numpy, Pandas, Matplotlib

## Code Structure
  The Main file is the MNISTNeural.py file which calls all the other functions from other files.
    At line 16 you can change data subsets from IID (load_data_IID) to non-IID (load_data_non_IID)
    At line 84 you can change the network topology between "Full", "Ring" and "Ring_Laplacian".

  At first run, the code will take a few minutes and then start showing messages on the output cmd interface to show that the training is going on.
  After training is over, the code should plot the test accuracy against the number of communication rounds.
