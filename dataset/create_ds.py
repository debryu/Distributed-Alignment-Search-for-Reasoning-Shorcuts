from pathlib import Path
import os
from torch import Tensor, load
import torch
import numpy as np
from tqdm import tqdm
from itertools import product
from random import sample, choice
from torchvision import datasets

def create_sample(X, target_sequence:tuple, digit2idx:dict, output_function:callable = np.sum):
    """Create a sample for a given sequence of digits.

    Keyword arguments:
      X: the dataset
      target_sequence: a list of which digits to use
      digit2idx: a dictionary in which each key [0-9] is associated with a list of indexes of the images with that label
      output_function: the function that will decide the output (default: np.sum)"""
    idxs = [choice(digit2idx[digit][0]) for digit in target_sequence]
    imgs = [X[idx] for idx in idxs]
    new_image = np.concatenate(imgs, axis=1) # Concatenate the images
    # The label will be the sequence of digits followed by the sum of the digits
    new_label = target_sequence + (output_function(target_sequence),)

    return new_image, np.array(new_label).astype('int32'), idxs


def create_dataset(n_digit:int = 2, 
                   sequence_len:int = 2, 
                   samples_x_world:int = 1000, 
                   train:bool = True, 
                   download:bool = False,
                   function:callable = np.sum,
                   custom_task:bool = False,
                   ) -> (np.ndarray, np.ndarray, dict):
    """Create a MNIST dataset with a given number of digits and sequence length.
        
        Keyword arguments:
          n_digit: the range of digits to use (default: 2)
          sequence_len: the length of the sequence of digits (default: 2)
          samples_x_world: the number of samples per world (default: 1000)
          train: whether to use the training set (default: True)
          download: whether to download the dataset (default: False)
          function: the function that will decide the output
          custom_task: whether to create a custom task dataset with particular constraints(default: False)
          """
    # Download data
    MNIST = datasets.MNIST(root='../data/', train=train, download=download)
    #print(train, samples_x_world, sequence_len, n_digit)
    x, y = MNIST.data, MNIST.targets

    # Create dictionary in which each key [0-9] is associated with a list of indexes of the images with that label
    digit2idx = {k: [] for k in range(10)}
    for k, v in digit2idx.items():
        v.append(np.where(y == k)[0])

    # Create the list of all possible permutations with repetition of 'sequence_len' digits
    if custom_task:
        worlds = list(product([0,1,2,3,4,6,7,8,9], repeat=sequence_len-1)) # All numbers except 5
        '''
        Edit the worlds to create the custom constraint where there is only one 5 
        If 5 is in position 0 or 2, the output will be 1
        If 5 is in position 1 or 3, the output will be 0
        The constraint is that if A+B > C+D then output is 1, 0 otherwise (where A is at index 0 and D is at index 3)

        There are 1458 possible worlds
        '''
        valid_worlds = []
        for sample in worlds:
            first,second,third = sample
            if first+5 > second+third:
                # The result should be 1 so 5 has to be in 0 position
                valid_worlds.append(tuple([5,first,second,third]))
            else:
                # The result should be 0 so 5 has to be in 1 position
                valid_worlds.append(tuple([first,5,second,third]))
            if first+second > third+5:
                # The result should be 1 so 5 has to be in 2 position
                valid_worlds.append(tuple([first,second,5,third]))
            else:
                # The result should be 0 so 5 has to be in 3 position
                valid_worlds.append(tuple([first,second,third,5]))

        worlds = list(valid_worlds)
    else:
        worlds = list(product(range(n_digit), repeat=sequence_len))

    imgs = [] # The list of all the samples
    labels = [] # The list of all the labels

    # Create data sample for each class
    for c in tqdm(worlds): # For every permutation...
        for i in range(samples_x_world): # ...create samples_x_world samples
            img, label, idxs = create_sample(x, c, digit2idx)
            imgs.append(img)
            labels.append(label)

    # Create dictionary of indexes for each combination (world)
    label2idx = {c: set() for c in worlds}
    for k, v in tqdm(label2idx.items()):
        for i, label in enumerate(labels): # For every label (one label contains a sequence)...
            if tuple(label[:sequence_len]) == k: #...if the label matches the current combination...
                v.add(i)  #...add the index to the set
    label2idx = {k: torch.tensor(list(v)) for k, v in label2idx.items()}

    return np.array(imgs).astype('int32'), np.array(labels), label2idx


def check_dataset(n_digits: int, 
                  sequence_len: int, 
                  data_folder: str, 
                  data_file: str, 
                  dataset_dim: dict,
                  function: callable = np.sum,
                  custom_task: bool = False) -> None:
    """Checks whether the dataset exists, if not creates it.
    
    Keyword arguments:
      n_digits: number of digits in the dataset
      sequence_len: length of the sequence of digits 
      data_folder: the path to the folder where the dataset is stored
      data_file: the name of the file where the dataset is stored
      dataset_dim: the dimensions of the dataset in a dictionary {train: int, test: int}
      function: the function that will decide the output (default: np.sum)
      custom_task: whether to create a custom task dataset with particular constraints(default: False)
    """
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(data_folder, data_file) + ".pt"
    print(f"Trying to load dataset located at {data_path}")
    try:
        load(data_path)
    except:
        print(f"No dataset found at {data_path}.")
        print(f'Saving to {data_folder}/{data_file}.pt')
        if custom_task:
            # Define dataset dimension so to have the same number of worlds
            n_worlds = (n_digits-1) ** (sequence_len-1) # The number of possible combinations, not considering the 5
            samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
            dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}

            train_imgs, train_labels, train_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                    samples_x_world=samples_x_world[f'{data_file}_train'], train=True,
                                                                    download=True, function=function, custom_task=custom_task)

            test_imgs, test_labels, test_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                samples_x_world=samples_x_world[f'{data_file}_test'], train=False,
                                                                download=True, function=function, custom_task=custom_task)
        else:
            n_worlds = (n_digits) ** (sequence_len) # The number of possible combinations, not considering the 5
            samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
            dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}

            train_imgs, train_labels, train_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                    samples_x_world=samples_x_world[f'{data_file}_train'], train=True,
                                                                    download=True, function=function,custom_task=custom_task)

            test_imgs, test_labels, test_indexes = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                samples_x_world=samples_x_world[f'{data_file}_test'], train=False,
                                                                download=True, function=function, custom_task=custom_task)

        print(f"Dataset dimensions: \n\t{dataset_dim[f'{data_file}_train']} train ({samples_x_world[f'{data_file}_train']} samples per world) \n\t{dataset_dim[f'{data_file}_test']} test ({samples_x_world[f'{data_file}_test']} samples per world)")
        
        data = {
                 f'{data_file}_train': {'images': train_imgs, 'labels': train_labels},
                 f'{data_file}_test': {'images': test_imgs, 'labels': test_labels},
               }

        indexes = {
                    data_file: train_indexes,
                    data_file: test_indexes
                  }

        torch.save(data, data_path)
        for key, value in indexes.items():
            torch.save(value, os.path.join(data_folder, f'{key}_indexes.pt'))

        print(f"Dataset saved in {data_folder}")