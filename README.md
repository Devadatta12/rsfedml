# rsfedml
A repository for application of fedarated machine learning techniques to remote sensing datasets.

The dataset we select is RSI-CB256 (link).

<b>rsfedml</b> is a software for the evaluation the application of federated learning to remote sensing datasets. The remote sensing dataset of choice is RSI-CB256. We also use the ResNet-50 model for image classification and Flower framework for provisioning the federated learning environment.

<h2> Dataset</h2>
<b>RSI-CB256</b> is the dataset of choice. However, any other dataset can be used if it follows the structure stated below:
<br><br>
data/dataset/train/class-category/png-images
data/dataset/test/class-category/png-images
<br><br>
data/federated_dataset/iid/train/partition_ID/class-category/png-images
data/federated_dataset/iid/test/partition_ID/class-category/png-images
<br><br>
data/federated_dataset/non_iid/train/partition_ID/class-category/png-images
data/federated_dataset/non_iid/test/partition_ID/class-category/png-images
<br><br>
data/federated_dataset/complete_non_iid/train/partition_ID/class-category/png-images
data/federated_dataset/complete_non_iid/test/partition_ID/class-category/png-images
<br><br>
Although the RSI-CB256 dataset does not have the above structure it can be transformed using the dataset.py.
Due to problems with partitioning using existing tools, we write our own partitioning logic in 'federated_dataset.py' to form the above 'iid' and 'non_iid' folders. 'iid' contains a dataset that has identical distribution on all its partitions. 'non_iid' contains a dataset that has some missing classes in all its partitions. 'complete_non_iid' contains a dataset that has all classes in all partitions, however the distribution is different across partitions. We use Dirilicht distribution to sample for each partition. <b> Few Manual efforts are required to reorganize the 'complete_non_iid' to guarantee the completeness property if some partitions may not have certain classes.</b> A solution to this problem is in progress. 


<h2> Code Structure</h2>
<ul>
<li> <b>data/</b> - Must contain the dataset in the required format under the folder. For the correct format, please refer to the Dataset section above. 
<li> <b>code/</b> - The code for remote sensing classification using Flower federated learning:
<ul>
<li> centralized (Centralized classification using ResNet model).
<li> fedavg (Federated learning classification using ResNet model and FedAvg aggregation algorithm).
<li> fedentropy (Federated learning classification using ResNet model and our FedEntropy aggregation algorithm).
<li> ferr (Federated learning classification using ResNet model and our FedSelect(Round Robin) aggregation algorithm).
<li> quickfedrandom (Federated learning classification using ResNet model and our FedSelect(Randomized) aggregation algorithm).
<li> quickfed (Federated learning classification using ResNet model and our FedSelect(Randomized+FedAvg) aggregation algorithm).
</ul>
<li> noniid (All the algorithms above, however on a non-iid dataset distribution).
</ul>

<h2> Installation Instructions :</h2>
<br>
<h3>Clone this repository:</h3>

```commandline
git clone https://github.com/Devadatta12/rsfedml.git
cd rsfedml
```
<br>

<h3> Requirements:</h3>
1) Python 10 or 11<br>

```commandline
python3.10 -m venv fedvenv
source fedvenv/bin/activate
```
<br>
4) Install python dependencies:

```commandline
pip install -r requirements.txt
```
<br>
<br>

<h3> Datasets:</h3>
<h4>Downloading Datasets:</h4>
The dataset that is tested in this repositiory is the RSI-CB256 dataset. Download the following dataset in the 'data/' directory and extract as a folder.
<ul>
<li> RSI-CB256: <a href="https://github.com/lehaifeng/RSI-CB"> Download images as .rar file from the Microsoft OneDrive in this link.</a>
</ul>

<h4>Generating IID and Non-IID Datasets:</h4>
To generate the folder structure suitable for Flower and Tensorflow run the following command.
````commandline
python3 dataset.py
````

Then generate the I.I.D and non-I.I.D datasets using the following command.
````commandline
python3 federated_dataset.py
````

The 'complete_non_iid' folder may not satisfy completeness requirements and needs to be manually checked.
You can contact us for receiving the iid and complete non-iid datasets via file transfer.

<br>
<h3> Launching:</h3>

<h4> Centralized Classification:</h4>

```commandline
cd code/classification/remote-sensing/centralized
python3 centralized.py
```

<h4> Federated Learning:</h4>
Following are the instructions for launching IID data experiments.

FedAvg:

```commandline
cd code/classification/remote-sensing/fedavg
flwr run
```

FedEntropy:

```commandline
cd code/classification/remote-sensing/fedentropy
flwr run
```

FedSelect (Round Robin): 

```commandline
cd code/classification/remote-sensing/fedrr
flwr run
```

FedSelect (Randomized):

```commandline
cd code/classification/remote-sensing/quickfedrandom
flwr run
```

FedSelect (Randomized + FedAvg):

```commandline
cd code/classification/remote-sensing/quickfed
flwr run
```



<h4> Non-IID Data Experiments</h4>
Following are the instructions for launching IID data experiments.

FedAvg:

```commandline
cd code/classification/remote-sensing/non_iid/fedavg
flwr run
```

FedEntropy:

```commandline
cd code/classification/remote-sensing/non_iid/fedentropy
flwr run
```

FedSelect (Round Robin): 

```commandline
cd code/classification/remote-sensing/non_iid/fedrr
flwr run
```

FedSelect (Randomized):

```commandline
cd code/classification/remote-sensing/non_iid/quickfedrandom
flwr run
```

FedSelect (Randomized + FedAvg):

```commandline
cd code/classification/remote-sensing/non_iid/quickfed
flwr run
```


Each experiment generates a list of accuracies at its completion.