import torch
from tqdm import tqdm
from models.alignment_model import compute_intervention, compute_intervention_sum

def compute_accuracy(backbone, rot_model, test_dl, layer, args):
  '''Compute IIA for the custom task models'''
  targ_layer = layer['layer']
  correct = 0
  total = 0
  correct_baseline = 0
  total_baseline = 0
  with torch.no_grad():
    for i, (imgs,labels,concepts) in enumerate(tqdm(test_dl)):
        img_base, img_source1, img_source2 = imgs
        img_base = img_base.to(args.device)
        img_source1 = img_source1.to(args.device)
        img_source2 = img_source2.to(args.device)

        if backbone.model_type == 'in-between':
          base,_,_,_,_,_,_,l01,l02,l03 = backbone(img_base)
          source1,_,_,_,_,_,_,l11,l12,l13 = backbone(img_source1)
          source2,_,_,_,_,_,_,l21,l22,l23 = backbone(img_source2)
        elif backbone.model_type == 'aligned':
          base,_,_,_,_,_,_,d0_l1,d0_l2,d0_l3,s0_l11, s0_l12, s0_l13, s0_l21, s0_l22, s0_l23 = backbone(img_base)
          source1,_,_,_,_,_,_,d1_l1,d1_l2,d1_l3,s1_l11, s1_l12, s1_l13, s1_l21, s1_l22, s1_l23 = backbone(img_source1)
          source2,_,_,_,_,_,_,d2_l1,d2_l2,d2_l3,s2_l11, s2_l12, s2_l13, s2_l21, s2_l22, s2_l23 = backbone(img_source2)
        elif backbone.model_type == 'joint':
          base,e01,e02,e03,d0_c,l01,l02,l03 = backbone(img_base)
          source1,e11,e12,e13,d1_c,l11,l12,l13 = backbone(img_source1)
          source2,e21,e22,e23,d2_c,l21,l22,l23 = backbone(img_source2)
        labels = torch.cat([labels],dim=0) # 16x3
        concepts = torch.cat([concepts],dim=0) # 16x3x4

        ''' 
        Assuming the high level model computes the sum

        Building all the interventions
        '''
        A,B,C,D = torch.split(img_base,28,dim=-1)
        
        E,F,G,H = torch.split(img_source1,28,dim=-1)
        I,L,M,N = torch.split(img_source2,28,dim=-1)
        test = torch.cat((A,B,E,F),dim=-1)
        test_1 = {'element':[0,1,0,1], 'source':[0,0,1,1]}
        lab_test = compute_intervention(concepts,test_1)

        interventions = []
        # Interventions on single digit
        interv_1 = torch.cat((A,B,C,G),dim=-1).to(args.device)
        st_1 = {'element':[0,1,2,2], 'source':[0,0,0,1]}
        interv_2 = torch.cat((A,B,H,D),dim=-1).to(args.device)
        st_2 = {'element':[0,1,3,3], 'source':[0,0,1,0]}
        interv_3 = torch.cat((A,E,C,D),dim=-1).to(args.device)
        st_3 = {'element':[0,0,2,3], 'source':[0,1,0,0]}
        interv_4 = torch.cat((F,B,C,D),dim=-1).to(args.device)
        st_4 = {'element':[1,1,2,3], 'source':[1,0,0,0]}

        # Interventions on the sum
        interv_5 = torch.cat((A,B,E,F),dim=-1).to(args.device)
        st_5 = {'element':[0,1,0,1], 'source':[0,0,1,1]}
        interv_6 = torch.cat((A,B,G,H),dim=-1).to(args.device)
        st_6 = {'element':[0,1,2,3], 'source':[0,0,1,1]}
        interv_7 = torch.cat((E,F,C,D),dim=-1).to(args.device)
        st_7 = {'element':[0,1,2,3], 'source':[1,1,0,0]}
        interv_8 = torch.cat((G,H,C,D),dim=-1).to(args.device)
        st_8 = {'element':[2,3,2,3], 'source':[1,1,0,0]}
        #interv_9 = torch.cat((A,B,M,N),dim=-1).to(args.device)

        # To detect the sum concept
        interventions = [
          #{'intervention':interv_1, 'labels': compute_intervention(concepts,st_1), 'index': 1}, # Index refers to the intervention location wrt the high level model
          #{'intervention':interv_2, 'labels': compute_intervention(concepts,st_2), 'index': 1},
          #{'intervention':interv_3, 'labels': compute_intervention(concepts,st_3), 'index': 0},
          #{'intervention':interv_4, 'labels': compute_intervention(concepts,st_4), 'index': 0},
          {'intervention':interv_5, 'labels': compute_intervention(concepts,st_5), 'index': 1},
          {'intervention':interv_6, 'labels': compute_intervention(concepts,st_6), 'index': 1},
          {'intervention':interv_7, 'labels': compute_intervention(concepts,st_7), 'index': 0},
          {'intervention':interv_8, 'labels': compute_intervention(concepts,st_8), 'index': 0}
        ]

        interv_single_digit1 = torch.cat((E,B,C,D),dim=-1).to(args.device)
        site_type_single_digit1 = {'element':[0,1,2,3], 'source':[1,0,0,0]}
        interv_single_digit2 = torch.cat((I,B,C,D),dim=-1).to(args.device)
        site_type_single_digit2 = {'element':[0,1,2,3], 'source':[2,0,0,0]}
        '''
        # To detect the single digit concept
        interventions = [
          {'intervention': interv_single_digit1, 'labels': compute_intervention(concepts,site_type_single_digit1), 'index': 0},
          {'intervention': interv_single_digit2, 'labels': compute_intervention(concepts,site_type_single_digit2), 'index': 0},
        ]
        '''
        for intervention in interventions:
          site_idx = intervention['index']
          if backbone.model_type == 'joint':
            if targ_layer == 'e1':
              intervened_base = rot_model(e01, e11, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_encoder1 = intervened_base)
            elif targ_layer == 'e2':
              intervened_base = rot_model(e02, e12, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_encoder2 = intervened_base)
            elif targ_layer == 'e3':
              intervened_base = rot_model(e03, e13, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_encoder3 = intervened_base)
            elif targ_layer == 'dc':
              intervened_base = rot_model(d0_c, d1_c, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_dense_c = intervened_base)
            elif targ_layer == 'l1':
              intervened_base = rot_model(l01, l11, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_layer1 = intervened_base)
            elif targ_layer == 'l2':
              intervened_base = rot_model(l02, l12, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_layer2 = intervened_base)
            elif targ_layer == 'l3':
              intervened_base = rot_model(l03, l13, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_ = backbone(img_base, intervention_layer3 = intervened_base)
          elif backbone.model_type == 'in-between':
            if targ_layer == 'l1':
              intervened_base = rot_model(l01, l11, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,l1,l2,l3 = backbone(img_base, intervention_layer1 = intervened_base)
            elif targ_layer == 'l2':
              intervened_base = rot_model(l02, l12, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,l1,l2,l3 = backbone(img_base, intervention_layer2 = intervened_base)
            elif targ_layer == 'l3':
              intervened_base = rot_model(l03, l13, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,l1,l2,l3 = backbone(img_base, intervention_layer3 = intervened_base)
          elif backbone.model_type == 'aligned':
            if targ_layer == 'l1':
              base_layer = torch.cat([s0_l11, s0_l21], dim=-1)
              source_layer = torch.cat([s1_l11, s1_l21], dim=-1)
              intervened_base = rot_model(base_layer, source_layer, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(img_base, sum_layer1 = intervened_base)
            elif targ_layer == 'l2':
              base_layer = torch.cat([s0_l12, s0_l22], dim=-1)
              source_layer = torch.cat([s1_l12, s1_l22], dim=-1)
              intervened_base = rot_model(base_layer, source_layer, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(img_base, sum_layer2 = intervened_base)
            elif targ_layer == 'l3':
              base_layer = torch.cat([s0_l13, s0_l23], dim=-1)
              source_layer = torch.cat([s1_l13, s1_l23], dim=-1)
              intervened_base = rot_model(base_layer, source_layer, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(img_base, sum_layer3 = intervened_base)
            elif targ_layer == 'd1':
              intervened_base = rot_model(d0_l1, d1_l1, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(img_base, dis_layer3 = intervened_base)
            elif targ_layer == 'd2':
              intervened_base = rot_model(d0_l2, d1_l2, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(img_base, dis_layer2 = intervened_base)
            elif targ_layer == 'd3':
              intervened_base = rot_model(d0_l3, d1_l3, site_idx = site_idx)
              intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(img_base, dis_layer3 = intervened_base)

          if backbone.model_type == 'in-between':
            baseline,_,_,_,_,_,_,_,_,_, = backbone(intervention['intervention'])
          elif backbone.model_type == 'aligned':
            baseline,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(intervention['intervention'])
          elif backbone.model_type == 'joint': 
            baseline,_,_,_,_,_,_,_ = backbone(intervention['intervention'])

          i_labels = intervention['labels']
          if targ_layer == 'base':
            intervened_output = base
            i_labels = labels[:,0]
          elif targ_layer == 'source1':
            intervened_output = source1
            i_labels = labels[:,1]
          elif targ_layer == 'source2':
            intervened_output = source2
            i_labels = labels[:,2]
          elif targ_layer == 'test':
            intervened_output,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = backbone(test)
            i_labels = lab_test
      
          pred_output = torch.argmax(intervened_output,dim=-1)
          baseline_output = torch.argmax(baseline,dim=-1)
          i_labels = i_labels.to(args.device)
          #print(pred_output)
          
          correct_baseline += torch.sum(baseline_output == i_labels)
          correct += torch.sum(pred_output == i_labels)
          #print(correct)  
          
          #print(len(i_labels))
        total += len(interventions)*len(i_labels)
        total_baseline += len(interventions)*len(i_labels)
        
  #print(total, len(test_dl), len(interventions),len(i_labels), len(i_labels)*len(interventions))
  print(f"Accuracy: {correct/total}")
  print(f"Baseline Accuracy: {correct_baseline/total_baseline}")

def compute_accuracy_sum_joint(backbone, rot_model, test_dl, layer, args):
  '''Compute IIA for MNIST-addition joint model'''
  targ_layer = layer['layer']
  correct = 0
  total = 0
  correct_baseline = 0
  total_baseline = 0
  with torch.no_grad():
    for i, (imgs,labels,concepts) in enumerate(tqdm(test_dl)):
        img_base, img_source1, img_source2 = imgs
        img_base = img_base.to(args.device)
        img_source1 = img_source1.to(args.device)
        img_source2 = img_source2.to(args.device)
        base,e01,e02,e03,d_c0,l01,l02,l03 = backbone(img_base)
        source1,e11,e12,e13,d_c1,l11,l12,l13 = backbone(img_source1)
        source2,e21,e22,e23,d_c2,l11,l12,l13 = backbone(img_source2)
        labels = torch.cat([labels],dim=0) # 16x3
        concepts = torch.cat([concepts],dim=0) # 16x3x2
        ''' 
        Assuming the high level model computes the sum

        Building all the interventions
        '''
        A,B, = torch.split(img_base,28,dim=-1)
        C,D = torch.split(img_source1,28,dim=-1)
        E,F= torch.split(img_source2,28,dim=-1)
       
        interventions = []
        # Interventions on single digit
        interv_1 = torch.cat((C,B),dim=-1).to(args.device)
        st_1 = {'element':[0,1], 'source':[1,0]}
        interventions = [
          {'intervention':interv_1, 'labels': compute_intervention_sum(concepts,st_1), 'index': 0}
        ]

        for intervention in interventions:
            site_idx = intervention['index']
            i_labels = intervention['labels']
            if targ_layer == 'e1':
                intervened_base = rot_model(e01, e11, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_encoder1 = intervened_base)
            elif targ_layer == 'e2':
                intervened_base = rot_model(e02, e12, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_encoder2 = intervened_base)
            elif targ_layer == 'e3':
                intervened_base = rot_model(e03, e13, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_encoder3 = intervened_base)
            elif targ_layer == 'dc':
                intervened_base = rot_model(d_c0, d_c1, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_dense_c = intervened_base)
            elif targ_layer == 'l1':
                intervened_base = rot_model(l01, l11, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_layer1 = intervened_base)
            elif targ_layer == 'l2':
                intervened_base = rot_model(l02, l12, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_layer2 = intervened_base)
            elif targ_layer == 'l3':
                intervened_base = rot_model(l03, l13, site_idx = site_idx)
                intervened_output,e1,e2,e3,d_c,l1,l2,l3 = backbone(img_base, intervention_layer3 = intervened_base)
            baseline,_,_,_,_,_,_,_ = backbone(intervention['intervention'])      
            pred_output = torch.argmax(intervened_output,dim=-1)
            baseline_output = torch.argmax(baseline,dim=-1)
            i_labels = i_labels.to(args.device)
            correct_baseline += torch.sum(baseline_output == i_labels)
            correct += torch.sum(pred_output == i_labels)
        total += len(interventions)*len(i_labels)
        total_baseline += len(interventions)*len(i_labels)
  print(f"Accuracy: {correct/total}")
  print(f"Baseline Accuracy: {correct_baseline/total_baseline}")

def compute_accuracy_sum_split(backbone, rot_model, test_dl, layer, args):
  '''Compute IIA for MNIST-addition split model'''
  targ_layer = layer['layer']
  correct = 0
  total = 0
  correct_baseline = 0
  total_baseline = 0
  with torch.no_grad():
    for i, (imgs,labels,concepts) in enumerate(tqdm(test_dl)):
        img_base, img_source1, img_source2 = imgs
        img_base = img_base.to(args.device)
        img_source1 = img_source1.to(args.device)
        img_source2 = img_source2.to(args.device)
        base,_,_,l01,l02,l03 = backbone(img_base)
        source1,_,_,l11,l12,l13 = backbone(img_source1)
        source2,_,_,l21,l22,l23 = backbone(img_source2)
        labels = torch.cat([labels],dim=0) # 16x3
        concepts = torch.cat([concepts],dim=0) # 16x3x2
        ''' 
        Assuming the high level model computes the sum

        Building all the interventions
        '''
        A,B, = torch.split(img_base,28,dim=-1)
        C,D = torch.split(img_source1,28,dim=-1)
        E,F= torch.split(img_source2,28,dim=-1)
       
        interventions = []
        # Interventions on single digit
        interv_1 = torch.cat((C,B),dim=-1).to(args.device)
        st_1 = {'element':[0,1], 'source':[1,0]}
        interventions = [
          {'intervention':interv_1, 'labels': compute_intervention_sum(concepts,st_1), 'index': 0}
        ]

        for intervention in interventions:
            site_idx = intervention['index']
            i_labels = intervention['labels']
            if targ_layer == 'l1':
                intervened_base = rot_model(l01, l11, site_idx = site_idx)
                intervened_output,_,_,l1,l2,l3 = backbone(img_base, intervention_layer1 = intervened_base)
            elif targ_layer == 'l2':
                intervened_base = rot_model(l02, l12, site_idx = site_idx)
                intervened_output,_,_,l1,l2,l3 = backbone(img_base, intervention_layer2 = intervened_base)
            elif targ_layer == 'l3':
                intervened_base = rot_model(l03, l13, site_idx = site_idx)
                intervened_output,_,_,l1,l2,l3 = backbone(img_base, intervention_layer3 = intervened_base)
            baseline,_,_,_,_,_ = backbone(intervention['intervention'])      
            pred_output = torch.argmax(intervened_output,dim=-1)
            baseline_output = torch.argmax(baseline,dim=-1)
            i_labels = i_labels.to(args.device)
            correct_baseline += torch.sum(baseline_output == i_labels)
            correct += torch.sum(pred_output == i_labels)
        total += len(interventions)*len(i_labels)
        total_baseline += len(interventions)*len(i_labels)
  print(f"Accuracy: {correct/total}")
  print(f"Baseline Accuracy: {correct_baseline/total_baseline}")