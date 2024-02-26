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
                   train:bool = True, 
                   download:bool = False,
                   samples_per_world:int = 2,
                   function:callable = np.sum,
                   custom_task:bool = False,
                   constraint:bool = True,
                   ) -> (np.ndarray, np.ndarray, dict):
    """Create a MNIST dataset with a given number of digits and sequence length.
        
        Keyword arguments:
          n_digit: the range of digits to use (default: 2)
          sequence_len: the length of the sequence of digits (default: 2)
          samples_x_world: the number of samples per world (default: 1000)
          train: whether to use the training set (default: True)
          download: whether to download the dataset (default: False)
          function: the function that will decide the output
          constraint: whether to create a custom task dataset with particular constraints(default: True)
          """
    # Download data
    MNIST = datasets.MNIST(root='../data/', train=train, download=download)
    #print(train, samples_x_world, sequence_len, n_digit)
    x, y = MNIST.data, MNIST.targets

    # Create dictionary in which each key [0-9] is associated with a list of indexes of the images with that label
    digit2idx = {k: [] for k in range(10)}
    for k, v in digit2idx.items():
        v.append(np.where(y == k)[0])

    if custom_task:
        if constraint:
            # Create the list of all possible permutations with repetition of 'sequence_len' digits
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
    else:
        worlds = list(product(range(n_digit), repeat=sequence_len)) # All possible combinations of the digits

    intervention_set_images = []
    intervention_set_labels = []
    # Create data sample for each class
    for c in tqdm(worlds): # For every permutation...
      for i in range(samples_per_world): # ...create samples_x_world samples
        s1 = choice(worlds)
        s2 = choice(worlds)
        img_base, label_base, _ = create_sample(x, c, digit2idx)
        img_source1, label_source1, _ = create_sample(x, s1, digit2idx)
        img_source2, label_source2, _ = create_sample(x, s2, digit2idx)
        intervention_set_images.append([img_base, img_source1, img_source2])
        intervention_set_labels.append([label_base, label_source1, label_source2])
        if not constraint:
            break # This is just to not overwhel the memory with too many samples since the unconstrained dataset is 10 times bigger


    return np.array(intervention_set_images).astype('int32'), np.array(intervention_set_labels)


def check_dataset_interventions(n_digits: int, 
                  sequence_len: int, 
                  data_folder: str, 
                  data_file: str, 
                  dataset_dim: dict,
                  function: callable = np.sum,
                  custom_task:bool = False,
                  constraint:bool = True,
                  ) -> None:
    """Checks whether the dataset exists, if not creates it.
    
    Keyword arguments:
      n_digits: number of digits in the dataset
      sequence_len: length of the sequence of digits 
      data_folder: the path to the folder where the dataset is stored
      data_file: the name of the file where the dataset is stored
      dataset_dim: the dimensions of the dataset in a dictionary {train: int, test: int}
      function: the function that will decide the output (default: np.sum)
      constraint: whether to create a custom task dataset with particular constraints(default: True)
                    for this task, the dataset will have always one and only one 5 in the sequence
    """
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(data_folder, data_file) + ".pt"
    print(f"Trying to load dataset located at {data_path}")
    try:
        load(data_path)
    except:
        if constraint:
            # Define dataset dimension so to have the same number of worlds
            n_worlds = (n_digits-1) ** (sequence_len-1) # The number of possible combinations, not considering the 5
            samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
            dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}


            train_imgs, train_labels = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                    samples_per_world=samples_x_world[f'{data_file}_train'],
                                                                    train=True,
                                                                    download=True, function=function,
                                                                    custom_task=custom_task)

            test_imgs, test_labels = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                samples_per_world=samples_x_world[f'{data_file}_test'],
                                                                train=False,
                                                                download=True, function=function,
                                                                custom_task=custom_task)
        else:
            # Define dataset dimension so to have the same number of worlds
            n_worlds = (n_digits) ** (sequence_len) # The number of possible combinations, not considering the 5
            samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
            dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}


            train_imgs, train_labels = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                    samples_per_world=samples_x_world[f'{data_file}_train'],
                                                                    train=True,
                                                                    download=True, function=function,
                                                                    custom_task=custom_task,
                                                                    constraint=constraint)

            test_imgs, test_labels = create_dataset(n_digit=n_digits, sequence_len=sequence_len,
                                                                samples_per_world=samples_x_world[f'{data_file}_test'],
                                                                train=False,
                                                                download=True, function=function,
                                                                custom_task=custom_task,
                                                                constraint=constraint)
        

        dataset_dim = {f'{data_file}_train': len(train_imgs), f'{data_file}_test': len(test_imgs)}
        print(f"Dataset dimensions: \n\t{dataset_dim[f'{data_file}_train']} train ({samples_x_world[f'{data_file}_train']} samples per world) \n\t{dataset_dim[f'{data_file}_test']} test ({samples_x_world[f'{data_file}_test']} samples per world)")
        
        data = {
                 f'{data_file}_train': {'images': train_imgs, 'labels': train_labels},
                 f'{data_file}_test': {'images': test_imgs, 'labels': test_labels},
               }

        torch.save(data, data_path)

        print(f"Dataset saved in {data_folder}")