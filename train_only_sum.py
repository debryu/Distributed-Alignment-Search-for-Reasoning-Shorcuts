from dataset.create_ds import check_dataset
from dataset.load_ds import load_data, nMNIST
from models.model import CUSTOM_OPERATION, ALIGNED_CUSTOM_OPERATION, ADDITION_JOINT, ADDITION_SPLIT
import torch
from argparse import Namespace
import os
from tqdm import tqdm
import numpy as np

args = Namespace()
args.data_location = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/'
args.data_folder = 'custom_task'
args.num_epochs = 1000000
args.batch_size = 128
args.num_workers = 1
args.eval_check = 10
args.file_name = 'das_summation_dataset_no_interventions'
args.sequence_len = 2
args.patience = 3
args.train_examples = 10000
args.test_examples = 10000
args.model_save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/models/"
args.saved_model_name = 'entangled'
args.custom_task = False
n_digits=10

args.num_images = args.sequence_len
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dimensions= {f'{args.file_name}_train':args.train_examples, f'{args.file_name}_test': args.test_examples}
data_folder = os.path.join(args.data_location, args.data_folder)

def sum_calculator(concepts: torch.Tensor):
    sum = concepts[:,0] + concepts[:,1]
    
    return sum

def accuracy(model,dl):
    correct = 0
    total = 0
    for imgs,labels,concepts in dl:
            if model.model_type == 'only_sum_joint':
                out, _,_,_,_,_,_ = model(imgs.to(args.device))
            else:
                out, _,_,_,_,_ = model(imgs.to(args.device))
            pred_output = torch.argmax(out,dim=-1)
            i_labels = labels.to(args.device)
            correct += torch.sum(pred_output == i_labels)
            total += len(i_labels)
    print(f"Accuracy: {correct/total}")
    return correct/total
  
def train_split(train_dl,test_dl,args):
    model = ADDITION_SPLIT(args = args)
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    patience = 0
    max_lr_decrease = 5
    best_accuracy = 0
    
    model.train()
    lr_decrease = 0
    for epoch in range(args.num_epochs):
      train_losses = []
      for imgs,labels,concepts in tqdm(train_dl):
          
          sum = sum_calculator(concepts)
          #rhs = rhs.type(torch.LongTensor).to(args.device)
          sum = sum.type(torch.LongTensor).to(args.device)
        
          out, _,_,_,_,_ = model(imgs.to(args.device))
          loss = criterion(out, sum)
          train_losses.append(loss.item())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          train_losses.append(loss.item())
      print(f"Epoch {epoch} loss: {np.mean(train_losses)}")

      if epoch % args.eval_check == 0:
        eval_losses = []
        model.eval()
        for imgs,labels,concepts in test_dl:
            # Concepts size: batch_size x 4 
            sum = sum_calculator(concepts)
            #rhs = rhs.type(torch.LongTensor).to(args.device)
            sum = sum.type(torch.LongTensor).to(args.device)
          
            out, _,_,_,_,_ = model(imgs.to(args.device))
            loss = criterion(out, sum)
            eval_losses.append(loss.item())
        avg_loss = np.mean(eval_losses)
        print(f"Epoch {epoch} test loss: {avg_loss}")
        acc = accuracy(model, test_dl)
        if acc > best_accuracy:
          patience = 0
          best_accuracy = acc
          torch.save(model.state_dict(), args.model_save_path + args.file_name + f"_best_{args.saved_model_name}_{epoch}.pt")
        else:
           patience += 1

        if patience > args.patience:
          print("Decreasing LR")
          learning_rate_scheduler.step()
          lr_decrease += 1
          patience = 0
          if lr_decrease > max_lr_decrease:
            break
            
def train_joint(train_dl,test_dl,args):
    model = ADDITION_JOINT(args = args)
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    patience = 0
    max_lr_decrease = 5
    best_accuracy = 0
    
    model.train()
    lr_decrease = 0
    for epoch in range(args.num_epochs):
      train_losses = []
      for imgs,labels,concepts in tqdm(train_dl):
          
          sum = sum_calculator(concepts)
          #rhs = rhs.type(torch.LongTensor).to(args.device)
          sum = sum.type(torch.LongTensor).to(args.device)
        
          out,_,_,_,_,_,_ = model(imgs.to(args.device))
          loss = criterion(out, sum)
          train_losses.append(loss.item())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          train_losses.append(loss.item())
      print(f"Epoch {epoch} loss: {np.mean(train_losses)}")

      if epoch % args.eval_check == 0:
        eval_losses = []
        model.eval()
        for imgs,labels,concepts in test_dl:
            # Concepts size: batch_size x 4 
            sum = sum_calculator(concepts)
            #rhs = rhs.type(torch.LongTensor).to(args.device)
            sum = sum.type(torch.LongTensor).to(args.device)
          
            out,_,_,_,_,_,_ = model(imgs.to(args.device))
            loss = criterion(out, sum)
            eval_losses.append(loss.item())
        avg_loss = np.mean(eval_losses)
        print(f"Epoch {epoch} test loss: {avg_loss}")
        acc = accuracy(model, test_dl)
        if acc > best_accuracy:
          patience = 0
          best_accuracy = acc
          torch.save(model.state_dict(), args.model_save_path + args.file_name + f"_best_{args.saved_model_name}_{epoch}.pt")
        else:
           patience += 1

        if patience > args.patience:
          print("Decreasing LR")
          learning_rate_scheduler.step()
          lr_decrease += 1
          patience = 0
          if lr_decrease > max_lr_decrease:
            break
def main():
    
    # Check whether dataset exists, if not build it
    check_dataset(n_digits, args.sequence_len, data_folder, args.file_name, dataset_dimensions, custom_task=args.custom_task)
    train_ds,test_ds = load_data(data_file=args.file_name, data_folder=data_folder, args=args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)


    #args.saved_model_name = 'disentangled'
    #train_split(train_dl,test_dl,args)
    args.saved_model_name = 'entangled'
    train_joint(train_dl,test_dl,args)
    
  
if __name__ == '__main__':
  main()