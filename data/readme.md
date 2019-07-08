To download the json file containing the url of the image data, please go to https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data.

After downloading the training.json file, you can run the 'delete_broken_images.py' to remove the broken images listed in the 'missing.json'. Then by running the 'download_images.py' you can download more than 1 million fashion images with tags to your local computer. Here, I put all data in the validation folder as I don't have seperated validation data. The validation data is taken from the training set.

```
python delete_broken_images.py
python scripts/download_images.py ./data/train.json ./data/validation
```
