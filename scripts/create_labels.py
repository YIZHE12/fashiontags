#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.preprocessing import MultiLabelBinarizer
import json
import numpy as np
import os
import pandas as pd
import sys


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Not enough arguments; exiting!')
        sys.exit(1)

    data_dir, out_dir = sys.argv[1:]

    with open(os.path.join(data_dir, 'train.json')) as f:
        train = json.load(f)
    with open(os.path.join(data_dir, 'validation.json')) as f:
        validation = json.load(f)
    with open(os.path.join(data_dir, 'test.json')) as f:
        test = json.load(f)

    # Merge annotations and filenames
    train_merge = pd.merge(pd.DataFrame(train['images']),
                           pd.DataFrame(train['annotations']),
                           on='imageId',
                           how='inner')
    train_merge['imageId'] = train_merge['imageId'].astype(np.uint32)
#    train_merge['labelId'] = train_merge['labelId'].astype(list)

    train_filenames = os.listdir(os.path.join(data_dir, 'train'))
    if train_filenames:
        train_filenames = [int(os.path.splitext(os.path.basename(f))[0])
                           for f in train_filenames]
        train_merge = train_merge[train_merge['imageId'].isin(train_filenames)]

    valid_merge = pd.merge(pd.DataFrame(validation['images']),
                           pd.DataFrame(validation['annotations']),
                           on='imageId',
                           how='inner')
    valid_merge['imageId'] = valid_merge['imageId'].astype(np.uint32)
#    valid_merge['labelId'] = valid_merge['labelId'].astype(list)

    test_merge = pd.DataFrame(test['images'])
    test_merge['imageId'] = test_merge['imageId'].astype(np.uint32)

    # Intermediate save
    train_merge.to_csv(os.path.join(out_dir, 'labels_train.csv'))
    valid_merge.to_csv(os.path.join(out_dir, 'labels_validation.csv'))
    test_merge.to_csv(os.path.join(out_dir, 'test.csv'))

    # Binarize labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_merge['labelId'])
    validation_labels = mlb.transform(valid_merge['labelId'])
    dummy_label_col = list(mlb.classes_)
    dummy_label_col = pd.DataFrame(columns=dummy_label_col)

    # Save to npy/csv
    dummy_label_col.to_csv(os.path.join(out_dir, 'dummy_label_col.csv'),
                           index=False)
    np.save(os.path.join(out_dir, 'labels_train.npy'), train_labels)
    np.save(os.path.join(out_dir, 'labels_validation.npy'), validation_labels)

    sys.exit(0)
