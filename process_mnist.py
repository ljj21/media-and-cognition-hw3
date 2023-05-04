# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             process_mnist.py
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================

import torch
import numpy as np

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_0', type=int, default=0, help='No. of class 0')
    parser.add_argument('--class_1', type=int, default=1, help='No. of class 1')
    parser.add_argument('--feat_dim', type=int, default=10, help='dim of features')
    parser.add_argument('--data_path', type=str, default='MNIST', help='the data path')
    parser.add_argument('--num_train', type=int, default=1000, help='the number of training data')
    parser.add_argument('--num_val', type=int, default=500, help='the number of validation data')

    opt = parser.parse_args()

    train_data, train_label = torch.load(os.path.join(opt.data_path, 'train.pt'))
    val_data, val_label = torch.load(os.path.join(opt.data_path, 'val.pt'))

    train_data, val_data = train_data.reshape(train_data.shape[0], -1).float().numpy(), val_data.reshape(val_data.shape[0], -1).float().numpy()
    train_label, val_label = train_label.numpy(), val_label.numpy()
    
    # TODO: compute the mean of train_data 
    data_mean = ???
    # TODO: compute the covariance matrix of train_data 
    data_cov = ???
    # TODO: compute the SVD decompositon of data_cov using np.linalg.svd
    u, sigma, vh = ???
    # TODO: using PCA to compress the dimensionality of the train_data and val_data after subtracting the mean vector
    train_data = ???
    val_data = ???

    train_data_0 = train_data[train_label == opt.class_0]
    train_data_1 = train_data[train_label == opt.class_1]
    val_data_0 = val_data[val_label == opt.class_0]
    val_data_1 = val_data[val_label == opt.class_1]

    train_data_2class = np.concatenate([train_data_0[:opt.num_train], train_data_1[:opt.num_train]], axis=0)
    val_data_2class = np.concatenate([val_data_0[:opt.num_val], val_data_1[:opt.num_val]], axis=0)

    np.save('MNIST/train', train_data_2class)
    np.save('MNIST/val', val_data_2class)
