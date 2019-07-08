# Download and clean the data
To download the json file containing the url of the image data, please go to https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data.

After you finished downloading the training.json file, you can run the 'delete_broken_images.py' to remove the broken images listed in the 'missing.json'. Then by running the 'download_images.py' you can download more than 1 million fashion images with tags to your local computer. The csv file contains the label names of the files. The raw json file doesn't contain the correct label name but instead only has labelId. You will also need to run the create_label.py file to convert the labelId to dummy variables. The reason to do this is that, each example can have different number of labels and different labels. In order to train the model regardless which label the example contains, I converted the label to a dummy variable. If the example has a certain label, the column representing that label will be set to '1', otherwise, if the example doesn't have a label, that column will be set to '0'. 

```
python delete_broken_images.py
python scripts/download_images.py ./data/train.json ./data/train
python scripts/create_label.py
```
