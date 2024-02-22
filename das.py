from dataset.create_ds_interventions import check_dataset_interventions
from models.alignment_model import train_rotation,eval_rotation, train_rotation_sum_joint,train_rotation_sum_split
from dataset.load_ds import load_data, nMNIST
import torch
from argparse import Namespace
import os
from models.model import CUSTOM_OPERATION, ALIGNED_CUSTOM_OPERATION, ADDITION_JOINT, ADDITION_SPLIT
from utils.iia import compute_accuracy, compute_accuracy_sum_joint, compute_accuracy_sum_split

args = Namespace()
args.data_location = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/'
args.data_folder = 'custom_task'
args.meaningful_interv = 10000
args.num_workers = 1
'''
This is the dataset to choose from
For the custom task, the name is:
  -> das_dataset_interventions

For the simple sum task, the name is:
  -> das_summation_dataset_interventions

But in principle, the name can be anything for another custom dataset
'''
#args.file_name = 'das_dataset_interventions'
args.file_name = 'das_summation_dataset_interventions'

args.lr = 1e-3
args.num_epochs = 6
args.train_examples = 44
args.test_examples = 10

'''
From testing, the batch size highly affects the training of the model
300 works well for in general, while with less it doesn't even train at all
'''
args.batch_size = 300

args.model_save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/models"
'''
This is the model to choose from

It has to be located inside args.model_save_path
'''
#args.model_name = 'das_dataset_no_interventions_best_full_aligned_architecture_t_8.pt'
#args.model_name = 'das_summation_dataset_no_interventions_best_disentangled_290.pt'
args.model_name = 'das_summation_dataset_no_interventions_best_entangled_460.pt'


# This is the high level model to choose from
# Currently used only for the custom task
args.high_level_model = [
  {'name': 'summation_1'},
  {'name': 'summation_2'},
]
n_digits=10

'''
Set this to True if you want to use a custom task
Otherwise will use the simple sum task
'''
if 'summation' in args.file_name.split('_'): 
  args.custom_task = False
else:
  args.custom_task = True



args.sequence_len = 4 if args.custom_task else 2
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dimensions= {f'{args.file_name}_train':args.train_examples, f'{args.file_name}_test': args.test_examples}
data_folder = os.path.join(args.data_location, args.data_folder)
args.num_images = args.sequence_len #Just because the models initilization needs this parameter

def main():
    
    # Check whether dataset exists, if not build it
    check_dataset_interventions(n_digits, args.sequence_len, data_folder, args.file_name, dataset_dimensions, custom_task=False)
    train_ds,test_ds = load_data(data_file=args.file_name, data_folder=data_folder, args=args)

    print(train_ds[0])

    # Load the model
    
    if not args.custom_task:
      #model = ADDITION_SPLIT(args = args).to(args.device)
      model = ADDITION_JOINT(args = args).to(args.device)
    else:
      # CHOOSE THE MODEL TO LOAD by uncommenting
      #model = CUSTOM_OPERATION(args).to(args.device)
      model = ALIGNED_CUSTOM_OPERATION(args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, args.model_name)))
    model.eval()
    train_inter_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_inter_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if model.model_type == 'in-between' or model.model_type == 'only_sum_split':
      targeted_layers = [ {'layer': 'l1', 'dim': 256},
                          {'layer': 'l2', 'dim': 256},
                          {'layer': 'l3', 'dim': 256} ]
    elif model.model_type == 'only_sum_joint':
      targeted_layers = [ {'layer': 'e1', 'dim': 2048},
                          {'layer': 'e2', 'dim': 1024},
                          {'layer': 'e3', 'dim': 512},
                          {'layer': 'l1', 'dim': 256},
                          {'layer': 'l2', 'dim': 256},
                          {'layer': 'l3', 'dim': 256} ]
    elif model.model_type == 'aligned':
      targeted_layers = [{'layer': 'l1', 'dim': 512},
                         {'layer': 'l2', 'dim': 512},
                         {'layer': 'l3', 'dim': 512},
                         {'layer': 'd1', 'dim': 512},
                         {'layer': 'd2', 'dim': 512},
                         {'layer': 'd3', 'dim': 512},]
    

    for targ_layer in targeted_layers:
      if model.model_type == 'only_sum_joint':
        rotation_model = train_rotation_sum_joint(model, train_inter_dl, args,interchange_dim=128, layer=targ_layer)
        # Compute accuracy
        print("Computing accuracy for layer: ", targ_layer)
        compute_accuracy_sum_joint(model, rotation_model, test_inter_dl, targ_layer, args)
      elif model.model_type == 'only_sum_split':
        rotation_model = train_rotation_sum_split(model, train_inter_dl, args,interchange_dim=128, layer=targ_layer)
        # Compute accuracy
        print("Computing accuracy for layer: ", targ_layer)
        compute_accuracy_sum_split(model, rotation_model, test_inter_dl, targ_layer, args)
      else:
        rotation_model = train_rotation(model, train_inter_dl, args,interchange_dim=64, layer=targ_layer)
        # Compute accuracy
        print("Computing accuracy for layer: ", targ_layer)
        compute_accuracy(model, rotation_model, test_inter_dl, targ_layer, args)
      
      
    asd
    eval_rotation(rotation_model, model, test_inter_dl, args)
if __name__ == '__main__':
  main()