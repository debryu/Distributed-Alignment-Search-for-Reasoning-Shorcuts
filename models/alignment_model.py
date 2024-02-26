import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_label(a,b,c,d:torch.tensor) -> torch.tensor:
    
    inputs_left = torch.stack([a,b],dim=1)
    inputs_right = torch.stack([c,d],dim=1)
    lhs = torch.sum(inputs_left,dim=1)
    rhs = torch.sum(inputs_right,dim=1)
    return torch.where(lhs>rhs, torch.tensor(1), torch.tensor(0))

def compute_intervention(concepts, site_type):
  e = site_type['element'] # in {0,1,2,3}
  s = site_type['source'] # in {0,1,2}
  # For example if e = [1,2,3,4] and s = [1,1,0,0]
  input_1 = concepts[:,s[0],e[0]]   # From the 1 source, the first element
  input_2 = concepts[:,s[1],e[1]]   # From the 1 source, the second element
  input_3 = concepts[:,s[2],e[2]]   # From the base, the third element
  input_4 = concepts[:,s[3],e[3]]   # From the base, the fourth element
  interv_labels = get_label(input_1, input_2, input_3, input_4)
  return interv_labels

def compute_intervention_sum(concepts, site_type):
  e = site_type['element'] # in {0,1,2,3}
  s = site_type['source'] # in {0,1,2}
  # For example if e = [1,2,3,4] and s = [1,1,0,0]
  input_1 = concepts[:,s[0],e[0]]   # From the 1 source, the first element
  input_2 = concepts[:,s[1],e[1]]   # From the 1 source, the second element
  inputs = torch.stack([input_1, input_2],dim=1)
  return torch.sum(inputs,dim=1)
       

def eval_rotation(rotation_model, model, dataloader, args, layer_dim = 256, interchange_dim = None):
  criterion = torch.nn.CrossEntropyLoss()
  rotation_model.eval()
  eval_losses = []
  for i, (imgs,labels,concepts) in enumerate(tqdm(dataloader)):
      img_base, img_source1, img_source2 = imgs
      img_base = img_base.to(args.device)
      img_source1 = img_source1.to(args.device)
      img_source2 = img_source2.to(args.device)

      base,_,_,_,_,l01,l02,l03 = model(img_base)
      source1,_,_,_,_,l11,l12,l13 = model(img_source1)
      source2,_,_,_,_,l21,l22,l23 = model(img_source2)

      labels = torch.cat([labels],dim=0) # 16x3
      concepts = torch.cat([concepts],dim=0) # 16x3x4
      #print(labels.shape)
      #print(concepts.shape)
      
      ''' 
      Assuming the high level model computes the sum

      Building the interventions
      '''
      A,B,C,D = torch.split(img_base,28,dim=-1)
      
      E,F,G,H = torch.split(img_source1,28,dim=-1)
      I,L,M,N = torch.split(img_source2,28,dim=-1)

      intervened_base_1 = torch.cat((A,B,E,F),dim=-1).to(args.device)
      intervened_base_2 = torch.cat((E,F,C,D),dim=-1).to(args.device)
      intervened_base_3 = torch.cat((A,B,I,L),dim=-1).to(args.device)
      intervened_base_4 = torch.cat((I,L,C,D),dim=-1).to(args.device)
      #print(intervened_base_1.shape)

      site = 'sum_0'
      site_type = site.split('_')[0]
      site_idx = int(site.split('_')[1])
      intervened_base = rotation_model(l01, l11, site_idx = site_idx)
      intervened_output,_,_,_,_,l1,l2,l3 = model(img_base, intervention_layer1 = intervened_base)
      
      interv_labels = compute_intervention(concepts, site_type, site_idx)
      interv_labels = interv_labels.type(torch.LongTensor).to(args.device)
      
      loss1 = criterion(intervened_output,interv_labels)
      

      site = 'sum_1'
      site_type = site.split('_')[0]
      site_idx = int(site.split('_')[1])
      intervened_base = rotation_model(l01, l11, site_idx = site_idx)
      intervened_output,_,_,_,_,l1,l2,l3 = model(img_base, intervention_layer1 = intervened_base)
      
      interv_labels = compute_intervention(concepts, site_type, site_idx)
      interv_labels = interv_labels.type(torch.LongTensor).to(args.device)

      loss2 = criterion(intervened_output,interv_labels)

      loss = (loss1 + loss2)/2
      eval_losses.append(loss.item())
  print(f'Eval Loss: {np.mean(eval_losses)}')
  return np.mean(eval_losses)

def train_rotation(model, dataloader, args, layer, interchange_dim = None):
  '''Train the rotation matrix for the intervention'''
  targ_layer = layer['layer']
  layer_dim = layer['dim']
  trainable_rot = DistributedInterchangeIntervention(layer_dim=layer_dim, interchange_dim = interchange_dim, args=args)
  trainable_rot = trainable_rot.to(args.device)
  trainable_rot.train()
  #optimizer = torch.optim.AdamW(trainable_rot.parameters(), lr=args.lr)
  optimizer = torch.optim.SGD(trainable_rot.parameters(), lr=args.lr)
  criterion = torch.nn.CrossEntropyLoss()

  

  for epoch in range(args.num_epochs):
    train_losses = []
    for i, (imgs,labels,concepts) in enumerate(tqdm(dataloader)):
        img_base, img_source1, img_source2 = imgs
        img_base = img_base.to(args.device)
        img_source1 = img_source1.to(args.device)
        img_source2 = img_source2.to(args.device)

        if model.model_type == 'in-between':
          base,_,_,_,_,_,_,l01,l02,l03 = model(img_base)
          source1,_,_,_,_,_,_,l11,l12,l13 = model(img_source1)
          source2,_,_,_,_,_,_,l21,l22,l23 = model(img_source2)
        elif model.model_type == 'aligned':
          base, c01, c02, c03, c04, sum0_1, sum0_2, d0_l1, d0_l2, d0_l3, s0_l11, s0_l12, s0_l13, s0_l21, s0_l22, s0_l23 = model(img_base)
          source1, c11, c12, c13, c14, sum1_1, sum1_2, d1_l1, d1_l2, d1_l3, s1_l11, s1_l12, s1_l13, s1_l21, s1_l22, s1_l23 = model(img_source1)
          source2, c21, c22, c23, c24, sum2_1, sum2_2, d2_l1, d2_l2, d2_l3, s2_l11, s2_l12, s2_l13, s2_l21, s2_l22, s2_l23 = model(img_source2)
        elif model.model_type == 'joint':
          base,e01,e02,e03,d0_c,l01,l02,l03 = model(img_base)
          source1,e11,e12,e13,d1_c,l11,l12,l13 = model(img_source1)
          source2,e21,e22,e23,d2_c,l21,l22,l23 = model(img_source2)


        
        labels = torch.cat([labels],dim=0) # 16x3
        concepts = torch.cat([concepts],dim=0) # 16x3x4
        #print(labels.shape)
        #print(concepts.shape)
        
        ''' 
        Assuming the high level model computes the sum

        Building the interventions
        '''
        # Base
        A,B,C,D = torch.split(img_base,28,dim=-1)
        # Source (multi-source)
        E,F,G,H = torch.split(img_source1,28,dim=-1)
        I,L,M,N = torch.split(img_source2,28,dim=-1)

        intervened_base_1 = torch.cat((E,F,C,D),dim=-1).to(args.device)
        site_type_1 = {'element':[0,1,2,3], 'source':[1,1,0,0]}
        intervened_base_2 = torch.cat((A,B,G,H),dim=-1).to(args.device)
        site_type_2 = {'element':[0,1,2,3], 'source':[0,0,1,1]}
        intervened_base_3 = torch.cat((I,L,C,D),dim=-1).to(args.device)
        site_type_3 = {'element':[0,1,2,3], 'source':[2,2,0,0]}
        intervened_base_4 = torch.cat((A,B,M,N),dim=-1).to(args.device)
        site_type_4 = {'element':[0,1,2,3], 'source':[0,0,2,2]}

        interv_single_digit1 = torch.cat((E,B,C,D),dim=-1).to(args.device)
        site_type_single_digit1 = {'element':[0,1,2,3], 'source':[1,0,0,0]}
        interv_single_digit2 = torch.cat((I,B,C,D),dim=-1).to(args.device)
        site_type_single_digit2 = {'element':[0,1,2,3], 'source':[2,0,0,0]}

        # To detect sum concept
        interventions = [
          {'intervention': intervened_base_1, 'labels': compute_intervention(concepts,site_type_1), 'index': 0},
          {'intervention': intervened_base_2, 'labels': compute_intervention(concepts,site_type_2), 'index': 1},
          {'intervention': intervened_base_3, 'labels': compute_intervention(concepts,site_type_3), 'index': 0},
          {'intervention': intervened_base_4, 'labels': compute_intervention(concepts,site_type_4), 'index': 1},
        ]
        '''
        # To detect the single digit concept
        interventions = [
          {'intervention': interv_single_digit1, 'labels': compute_intervention(concepts,site_type_single_digit1), 'index': 0},
          {'intervention': interv_single_digit2, 'labels': compute_intervention(concepts,site_type_single_digit2), 'index': 0},
        ]
        '''
        losses = [] 
        
        for intervention in interventions:   # Do the intervention for each site depending on the high level model
          i_input = intervention['intervention']
          i_labels = intervention['labels']
          site_idx = intervention['index']
          #source_output,_,_,_,_,_,_,ls1,ls2,ls3 = model(i_input)
          #l01 = torch.zeros((16,2)).to(args.device)
          #l11 = torch.ones((16,2)).to(args.device)
          if model.model_type == 'joint':
            if targ_layer == 'e1':
              intervened_base = trainable_rot(e01, e11, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_encoder1 = intervened_base)
            elif targ_layer == 'e2':
              intervened_base = trainable_rot(e02, e12, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_encoder2 = intervened_base)
            elif targ_layer == 'e3':
              intervened_base = trainable_rot(e03, e13, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_encoder3 = intervened_base)
            elif targ_layer == 'dc':
              intervened_base = trainable_rot(d0_c, d1_c, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_dense_c = intervened_base)
            elif targ_layer == 'l1':
              intervened_base = trainable_rot(l01, l11, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_layer1 = intervened_base)
            elif targ_layer == 'l2':
              intervened_base = trainable_rot(l02, l12, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_layer2 = intervened_base)
            elif targ_layer == 'l3':
              intervened_base = trainable_rot(l03, l13, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = model(img_base, intervention_layer3 = intervened_base)
          if model.model_type == 'in-between':
            if targ_layer == 'l1':
              intervened_base = trainable_rot(l01, l11, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,l1,l2,l3 = model(img_base,intervention_layer1 = intervened_base)
            elif targ_layer == 'l2':
              intervened_base = trainable_rot(l02, l12, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,l1,l2,l3 = model(img_base, intervention_layer2 = intervened_base)
            elif targ_layer == 'l3':
              intervened_base = trainable_rot(l03, l13, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,l1,l2,l3 = model(img_base, intervention_layer3 = intervened_base)
          elif model.model_type == 'aligned':
            if targ_layer == 'l1':
              base_layer = torch.cat([s0_l11, s0_l21], dim=-1)
              source_layer = torch.cat([s1_l11, s1_l21], dim=-1)
              intervened_base = trainable_rot(base_layer, source_layer, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(img_base, sum_layer1 = intervened_base)
            elif targ_layer == 'l2':
              base_layer = torch.cat([s0_l12, s0_l22], dim=-1)
              source_layer = torch.cat([s1_l12, s1_l22], dim=-1)
              intervened_base = trainable_rot(base_layer, source_layer, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(img_base, sum_layer2 = intervened_base)
            elif targ_layer == 'l3':
              base_layer = torch.cat([s0_l13, s0_l23], dim=-1)
              source_layer = torch.cat([s1_l13, s1_l23], dim=-1)
              intervened_base = trainable_rot(base_layer, source_layer, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(img_base, sum_layer3 = intervened_base)
            elif targ_layer == 'd1':
              intervened_base = trainable_rot(d0_l1, d1_l1, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(img_base, dis_layer1 = intervened_base)
            elif targ_layer == 'd2':
              intervened_base = trainable_rot(d0_l2, d1_l2, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(img_base, dis_layer2 = intervened_base)
            elif targ_layer == 'd3':
              intervened_base = trainable_rot(d0_l3, d1_l3, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = model(img_base, dis_layer3 = intervened_base)
            else:
              intervened_output = base
            
          
          interv_labels = i_labels.type(torch.LongTensor).to(args.device)

          loss = criterion(intervened_output,interv_labels)
          train_losses.append(loss.item())
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()


    print(f'Epoch: {epoch} Loss: {np.mean(train_losses)}')
  return trainable_rot

def train_rotation_sum_joint(model, dataloader, args, layer, interchange_dim = None):
  '''Train the rotation matrix for the intervention, for the joint model (entangled)'''
  targ_layer = layer['layer']
  layer_dim = layer['dim']
  trainable_rot = DistributedInterchangeIntervention(layer_dim=layer_dim, interchange_dim = interchange_dim, args=args)
  trainable_rot = trainable_rot.to(args.device)
  trainable_rot.train()
  optimizer = torch.optim.AdamW(trainable_rot.parameters(), lr=args.lr)
  criterion = torch.nn.CrossEntropyLoss()
  for epoch in range(args.num_epochs):
    train_losses = []
    for i, (imgs,labels,concepts) in enumerate(dataloader):
        img_base, img_source1, img_source2 = imgs
        img_base = img_base.to(args.device)
        img_source1 = img_source1.to(args.device)
        img_source2 = img_source2.to(args.device)
        labels = torch.cat([labels],dim=0) # 16x3
        concepts = torch.cat([concepts],dim=0) # 16x3x2        
        ''' 
        Assuming the high level model computes the sum

        Building the interventions
        '''
        base,e01,e02,e03,d_c0,l01,l02,l03 = model(img_base)
        source1,e11,e12,e13,d_c1,l11,l12,l13 = model(img_source1)
        source2,e21,e22,e23,d_c2,l21,l22,l23 = model(img_source2)
        # Base
        A,B = torch.split(img_base,28,dim=-1)
        # Source (multi-source)
        C,D = torch.split(img_source1,28,dim=-1)
        E,F = torch.split(img_source2,28,dim=-1)
      
        #plot the images
        #img = torch.cat([img_base.squeeze(0).permute(1,2,0),img_source1.squeeze(0).permute(1,2,0),img_source2.squeeze(0).permute(1,2,0)],dim = 0).cpu().detach().numpy()
        #plt.imshow(img)
        #plt.show()
  
        interv_1 = torch.cat((C,B),dim=-1).to(args.device)
        site_type_1 = {'element':[0,1], 'source':[1,0]} # First element from source 1, second from base
        
        interventions = [
          {'intervention': interv_1, 'labels': compute_intervention_sum(concepts,site_type_1), 'index': 0},
        ]
        losses = [] 
        for intervention in interventions:   # Do the intervention for each site depending on the high level model
          i_input = intervention['intervention']
          i_labels = intervention['labels']
          site_idx = intervention['index']
          if targ_layer == 'e1':
            intervened_base = trainable_rot(e01, e11, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_encoder1 = intervened_base)
          elif targ_layer == 'e2':
            intervened_base = trainable_rot(e02, e12, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_encoder2 = intervened_base)
          elif targ_layer == 'e3':
            intervened_base = trainable_rot(e03, e13, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_encoder3 = intervened_base)
          elif targ_layer == 'dc':
            intervened_base = trainable_rot(d_c0, d_c1, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_dense_c = intervened_base)
          elif targ_layer == 'l1':
            intervened_base = trainable_rot(l01, l11, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_layer1 = intervened_base)
          elif targ_layer == 'l2':
            intervened_base = trainable_rot(l02, l12, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_layer2 = intervened_base)
          elif targ_layer == 'l3':
            intervened_base = trainable_rot(l03, l13, site_idx = site_idx)
            intervened_output,e1,e2,e3,d_c,l1,l2,l3 = model(img_base, intervention_layer3 = intervened_base)
          interv_labels = i_labels.type(torch.LongTensor).to(args.device)

          '''Optional code to visualize the effect of the intervention wrt the prediction'''
          '''
          if i==0:
            for i,a in enumerate(intervened_output[0]):
              print(f' {i}: {a}')
            print('Answer should be:' , interv_labels[0].item())
          '''
          loss = criterion(intervened_output,interv_labels)
          train_losses.append(loss.item())
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
    #print(f'Epoch: {epoch} Loss: {np.mean(train_losses)}')
  return trainable_rot

def train_rotation_sum_split(model, dataloader, args, layer, interchange_dim = None):
  '''Train the rotation matrix for the intervention, for the split model (disentangled)'''
  targ_layer = layer['layer']
  layer_dim = layer['dim']
  trainable_rot = DistributedInterchangeIntervention(layer_dim=layer_dim, interchange_dim = interchange_dim, args=args)
  trainable_rot = trainable_rot.to(args.device)
  trainable_rot.train()
  optimizer = torch.optim.AdamW(trainable_rot.parameters(), lr=args.lr)
  criterion = torch.nn.CrossEntropyLoss()
  for epoch in range(args.num_epochs):
    train_losses = []
    for i, (imgs,labels,concepts) in enumerate(dataloader):
        img_base, img_source1, img_source2 = imgs
        img_base = img_base.to(args.device)
        img_source1 = img_source1.to(args.device)
        img_source2 = img_source2.to(args.device)
        labels = torch.cat([labels],dim=0) # 16x3
        concepts = torch.cat([concepts],dim=0) # 16x3x2        
        ''' 
        Assuming the high level model computes the sum

        Building the interventions
        '''
        base,_,_,l01,l02,l03 = model(img_base)
        source1,_,_,l11,l12,l13 = model(img_source1)
        source2,_,_,l21,l22,l23 = model(img_source2)
        # Base
        A,B = torch.split(img_base,28,dim=-1)
        # Source (multi-source)
        C,D = torch.split(img_source1,28,dim=-1)
        E,F = torch.split(img_source2,28,dim=-1)
      
        #plot the images
        #img = torch.cat([img_base.squeeze(0).permute(1,2,0),img_source1.squeeze(0).permute(1,2,0),img_source2.squeeze(0).permute(1,2,0)],dim = 0).cpu().detach().numpy()
        #plt.imshow(img)
        #plt.show()
  
        interv_1 = torch.cat((C,B),dim=-1).to(args.device)
        site_type_1 = {'element':[0,1], 'source':[1,0]} # First element from source 1, second from base
        
        interventions = [
          {'intervention': interv_1, 'labels': compute_intervention_sum(concepts,site_type_1), 'index': 0},
        ]
        losses = [] 
        for intervention in interventions:   # Do the intervention for each site depending on the high level model
          i_input = intervention['intervention']
          i_labels = intervention['labels']
          site_idx = intervention['index']
          if targ_layer == 'l1':
            intervened_base = trainable_rot(l01, l11, site_idx = site_idx)
            intervened_output,_,_,l1,l2,l3 = model(img_base, intervention_layer1 = intervened_base)
          elif targ_layer == 'l2':
            intervened_base = trainable_rot(l02, l12, site_idx = site_idx)
            intervened_output,_,_,l1,l2,l3 = model(img_base, intervention_layer2 = intervened_base)
          elif targ_layer == 'l3':
            intervened_base = trainable_rot(l03, l13, site_idx = site_idx)
            intervened_output,_,_,l1,l2,l3 = model(img_base, intervention_layer3 = intervened_base)
          interv_labels = i_labels.type(torch.LongTensor).to(args.device)

          '''Optional code to visualize the effect of the intervention wrt the prediction'''
          '''
          if i==0:
            for i,a in enumerate(intervened_output[0]):
              print(f' {i}: {a}')
            print('Answer should be:' , interv_labels[0].item())
          '''
          loss = criterion(intervened_output,interv_labels)
          train_losses.append(loss.item())
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
    print(f'Epoch: {epoch} Loss: {np.mean(train_losses)}')
  return trainable_rot

class Rotation(torch.nn.Module):
  def __init__(self, dim , init_orth = True):
    super(Rotation, self).__init__()
    self.dim = dim
    weights = torch.empty(dim, dim)
    if init_orth:
      weight = torch.nn.init.orthogonal_(weights)
    self.weight = torch.nn.Parameter(weight, requires_grad=True)

  def forward(self, x):
    return torch.matmul(x, self.weight)

class DistributedInterchangeIntervention(torch.nn.Module):
  def __init__(self, layer_dim, args, interchange_dim = None, init_orth = True):
    super(DistributedInterchangeIntervention, self).__init__()
    self.rotation_matrix = Rotation(layer_dim, init_orth)
    self.rotation_operator = torch.nn.utils.parametrizations.orthogonal(self.rotation_matrix)
    self.layer_dim = layer_dim
    if interchange_dim is None:
      self.interchange_dim = (layer_dim//len(self.args.high_level_model)) - 1
    else:
      self.interchange_dim = interchange_dim
    self.args = args


  def forward(self, base, source, site_idx):
    #print(base.shape, source.shape)
    #assert base.shape == source.shape, "Base and source must have the same shape"
    #print(base.shape[-1], self.layer_dim)
    #assert base.shape[-1] == self.layer_dim, "Base and source must have the same shape"

    rotated_base = self.rotation_operator(base)
    #print('base',rotated_base)
    rotated_source = self.rotation_operator(source)
    #print('sorce',rotated_source)
    #print(self.layer_dim, self.interchange_dim)
    assert self.layer_dim % self.interchange_dim == 0, "interchange_dim should divide layer_dim evenly"
    
    rotated_base = rotated_base.reshape(-1, self.layer_dim // self.interchange_dim, self.interchange_dim)
    rotated_source = rotated_source.reshape(-1, self.layer_dim // self.interchange_dim, self.interchange_dim)
    # Do the interchange
    rotated_base[:,site_idx,:] = rotated_source[:,site_idx,:]
    # Reshape
    rotated_base = rotated_base.reshape(-1, self.layer_dim)
    
    new_base = torch.matmul(rotated_base, self.rotation_operator.weight.T)  # Orth matrix inverse is the transpose
    #print('new_base',new_base)
    return new_base