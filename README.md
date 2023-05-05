#NIP: An Efficient and Accurate Differentially Private Synthetic Data Scheme

NIP: An Efficient and Accurate Differentially Private Synthetic Data Scheme. NIP is a differentially private data synthesis scheme that takes a tabular dataset as input and outputs a synthetic dataset in the same format. 

The code can be executed in Python 3.6.9.

The dependencies of this project can be installed with pip as follows:

$ pip install -r requirements.txt

Additionally, you have to add the src folder to the PYTHONPATH. If you are using Ubuntu, add the following line to your .bashrc file:
PYTHONPATH=$PYTHONPATH:/path/to/NIP/src

Once this is done, test the scheme by the following code:
$ python NIP.py