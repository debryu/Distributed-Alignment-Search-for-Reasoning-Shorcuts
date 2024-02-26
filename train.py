from dataset.create_ds import check_dataset
from dataset.load_ds import load_data, nMNIST
from models.model import CUSTOM_OPERATION, ALIGNED_CUSTOM_OPERATION, JOINT_CUSTOM_OPERATION
import torch
from argparse import Namespace
import os
from tqdm import tqdm
import numpy as np

args = Namespace()
args.data_location = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/'
args.data_folder = 'custom_task'
args.num_epochs = 100
args.batch_size = 128
args.num_workers = 1
args.eval_check = 7
args.file_name = 'das_dataset_no_constraints_no_interventions'
args.constraint = False
args.sequence_len = 4
args.train_examples = 10000
args.test_examples = 10000
args.model_save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/models/"
args.saved_model_name = 'unconstrained'
args.custom_task = True # Set this to True if you want to use the custom task, or False if you want to use the simple sum task
'''
If you choose the custom task,
choose the model type between 'in-between', 'aligned' and 'joint'
'''
args.model_type = 'in-between'
args.fine_tune = False # Only used for the aligned model, to further fine tune it for the custom task
args.trained_model_to_fine_tune = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/models/das_dataset_no_interventions_best_full_aligned_architecture_28.pt"

n_digits=10

args.num_images = args.sequence_len
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dimensions= {f'{args.file_name}_train':args.train_examples, f'{args.file_name}_test': args.test_examples}
data_folder = os.path.join(args.data_location, args.data_folder)

def sum_calculator(concepts: torch.Tensor):
    lhs = concepts[:,0] + concepts[:,1]
    rhs = concepts[:,2] + concepts[:,3]
    return lhs, rhs

def accuracy(model,dl):
    correct = 0
    total = 0
    for imgs,labels,concepts in dl:
            # Concepts size: batch_size x 4 
            lhs, rhs = sum_calculator(concepts)
            rhs = rhs.type(torch.LongTensor).to(args.device)
            lhs = lhs.type(torch.LongTensor).to(args.device)
            concepts = concepts.type(torch.LongTensor).to(args.device).reshape(-1)

            if model.model_type == 'in-between':
                out,s_l,s_r,c1,c2,c3,c4,_,_,_ = model(imgs.to(args.device))
            elif model.model_type == 'aligned':
                out, c1, c2, c3, c4, s_l, s_r, _,_,_,_,_,_,_,_,_ = model(imgs.to(args.device)) 
            elif model.model_type == 'joint':
                out,e1,e2,e3,d_c,l1,l2,l3 = model(imgs.to(args.device))
            pred_output = torch.argmax(out,dim=-1)
            i_labels = labels.to(args.device)
            correct += torch.sum(pred_output == i_labels)
            total += len(i_labels)
    print(f"Accuracy: {correct/total}")
    return correct/total
  
            
def main():
    
    # Check whether dataset exists, if not build it
    check_dataset(n_digits, args.sequence_len, data_folder, args.file_name, dataset_dimensions, custom_task=True, constraint=args.constraint)
    train_ds,test_ds = load_data(data_file=args.file_name, data_folder=data_folder, args=args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.model_type == 'in-between':
      model = CUSTOM_OPERATION(args).to(args.device)  # aka in-between
    elif args.model_type == 'joint':
      model = JOINT_CUSTOM_OPERATION(args).to(args.device) # aka joint
    elif args.model_type == 'aligned':
      model = ALIGNED_CUSTOM_OPERATION(args).to(args.device) # aka aligned
      # Use this 2 lines only to further fine tune it
      if args.fine_tune:
        model.load_state_dict(torch.load(args.trained_model_to_fine_tune))
        optimizer = torch.optim.AdamW(model.diseq.parameters(), lr=0.0001)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    

    patience = 0
    best_loss = 1000
    
    model.train()
    for epoch in range(args.num_epochs):
      train_losses = []
      for imgs,labels,concepts in tqdm(train_dl):
          
          # Concepts size: batch_size x 4 
          lhs, rhs = sum_calculator(concepts)
          rhs = rhs.type(torch.LongTensor).to(args.device)
          lhs = lhs.type(torch.LongTensor).to(args.device)
          concepts = concepts.type(torch.LongTensor).reshape(-1).to(args.device)

          if model.model_type == 'in-between':
              out,s_l,s_r,c1,c2,c3,c4,_,_,_ = model(imgs.to(args.device))
              c_out = torch.cat([c1,c2,c3,c4],dim=1)
              c_out = c_out.reshape(-1,10) #batch_size x 4 x 10 -> [ ,10]
              digit_recog_loss = criterion(c_out, concepts)
              sum_lhs_loss = criterion(s_l, lhs)
              sum_rhs_loss = criterion(s_r, rhs)
              total_sum_loss = sum_lhs_loss + sum_rhs_loss
          elif model.model_type == 'aligned':
              out, c1, c2, c3, c4, s_l, s_r, _,_,_,_,_,_,_,_,_ = model(imgs.to(args.device))
              c_out = torch.cat([c1,c2,c3,c4],dim=1)
              c_out = c_out.reshape(-1,10) #batch_size x 4 x 10 -> [ ,10]
              digit_recog_loss = criterion(c_out, concepts)
              sum_lhs_loss = criterion(s_l, lhs)
              sum_rhs_loss = criterion(s_r, rhs)
              total_sum_loss = sum_lhs_loss + sum_rhs_loss
          elif model.model_type == 'joint':
              out,e1,e2,e3,d_c,l1,l2,l3 = model(imgs.to(args.device))
          
          labels = labels.type(torch.LongTensor).to(args.device)
          loss = criterion(out, labels)#total_sum_loss*digit_recog_loss#criterion(out, labels) + total_sum_loss*1000 + digit_recog_loss*10
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          train_losses.append(loss.item())
      print(f"Epoch {epoch} loss: {np.mean(train_losses)}")

      if epoch % args.eval_check == 0:
        eval_losses = []
        digit_recog_losses = []
        sum_losses = []
        task_losses = []
        model.eval()
        for imgs,labels,concepts in test_dl:
            # Concepts size: batch_size x 4 
            lhs, rhs = sum_calculator(concepts)
            rhs = rhs.type(torch.LongTensor).to(args.device)
            lhs = lhs.type(torch.LongTensor).to(args.device)
            concepts = concepts.type(torch.LongTensor).to(args.device).reshape(-1)
            if model.model_type == 'in-between':
                out,s_l,s_r,c1,c2,c3,c4,_,_,_ = model(imgs.to(args.device))
                c_out = torch.cat([c1,c2,c3,c4],dim=1)
                c_out = c_out.reshape(-1,10) #batch_size x 4 x 10 -> [ ,10]
                digit_recog_loss = criterion(c_out, concepts)
                sum_lhs_loss = criterion(s_l, lhs)
                sum_rhs_loss = criterion(s_r, rhs)
                total_sum_loss = sum_lhs_loss + sum_rhs_loss
                digit_recog_losses.append(digit_recog_loss.item())
                sum_losses.append(total_sum_loss.item())
            elif model.model_type == 'aligned':
                out, c1, c2, c3, c4, s_l, s_r, _,_,_,_,_,_,_,_,_ = model(imgs.to(args.device))
                c_out = torch.cat([c1,c2,c3,c4],dim=1)
                c_out = c_out.reshape(-1,10) #batch_size x 4 x 10 -> [ ,10]
                digit_recog_loss = criterion(c_out, concepts)
                sum_lhs_loss = criterion(s_l, lhs)
                sum_rhs_loss = criterion(s_r, rhs)
                total_sum_loss = sum_lhs_loss + sum_rhs_loss
                digit_recog_losses.append(digit_recog_loss.item())
                sum_losses.append(total_sum_loss.item())
            elif model.model_type == 'joint':
                out,e1,e2,e3,d_c,l1,l2,l3 = model(imgs.to(args.device))
           
            
        
            labels = labels.type(torch.LongTensor).to(args.device)
            #print(out.shape, labels.shape)
            task_loss = criterion(out, labels)
            task_losses.append(task_loss.item())
            loss = criterion(out, labels)#total_sum_loss #task_loss + digit_recog_loss*10 + total_sum_loss*10
            eval_losses.append(loss.item())
        avg_loss = np.mean(eval_losses)
        print(f"Epoch {epoch} test loss: {avg_loss}.\nDigit recog: {np.mean(digit_recog_losses)} - Sum: {np.mean(sum_losses)} - Task: {np.mean(task_losses)}")
        accuracy(model, test_dl)
        if avg_loss < best_loss:
          patience = 0
          best_loss = avg_loss
          torch.save(model.state_dict(), args.model_save_path + args.file_name + f"_best_{args.saved_model_name}_{epoch}.pt")
        else:
           patience += 1

        if patience > 3:
          print("Early stopping")
          break
  
if __name__ == '__main__':
  main()