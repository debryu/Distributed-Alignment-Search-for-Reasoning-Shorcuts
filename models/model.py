import torch
from argparse import Namespace

class ENCODER_MODULE(torch.nn.Module):
  def __init__(self, hid_channels = 10, c_dim = 10):
    super(ENCODER_MODULE, self).__init__()
    self.encoder1 = torch.nn.Sequential(
      torch.nn.Conv2d(1, hid_channels, kernel_size=4, stride=2, padding=3), # 16,16
      torch.nn.ReLU()
    )
    self.encoder2 = torch.nn.Sequential(
      torch.nn.Conv2d(hid_channels, hid_channels*4, kernel_size=4, stride=2, padding=1), # 8,8
      torch.nn.ReLU()
    )
    self.encoder3 = torch.nn.Sequential(
      torch.nn.Conv2d(hid_channels*4, hid_channels*16, kernel_size=4, stride=2, padding=1), # 4,4
      torch.nn.ReLU()
    )
    self.dropout = torch.nn.Dropout(p=0.4)
    self.dense_c = torch.nn.Linear(hid_channels*16*4*4, c_dim)
    

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.encoder1(x)
    x = self.dropout(x)
    #print(x.shape)
    first_layer = x.reshape(batch_size,-1)
    #print('first_layer',first_layer.shape)
    x = self.encoder2(x)
    x = self.dropout(x)
    #print(x.shape)
    second_layer = x.reshape(batch_size,-1)
    #print('second_layer',second_layer.shape)
    x = self.encoder3(x)
    #print(x.shape)
    third_layer = x.reshape(batch_size,-1)
    #print('third_layer',third_layer.shape)
    output = self.dense_c(third_layer).unsqueeze(1)
    # Same as unsqueeze(1)?
    #c = torch.stack(torch.split(output,      20,      dim=-1), dim=1)
    
    return output
        
class MLP_MODULE(torch.nn.Module):
  def __init__(self, c_dim = 10, hidden = 256, num_images = 2):
    super(MLP_MODULE, self).__init__()
    self.layer1 = torch.nn.Linear(c_dim*num_images, hidden)
    self.layer2 = torch.nn.Linear(hidden, hidden)
    self.layer3 = torch.nn.Linear(hidden, hidden)
    self.classifier = torch.nn.Linear(hidden, c_dim*2-1)
    self.classifier2 = torch.nn.Linear(hidden, c_dim*2-1)
    self.output = torch.nn.Linear(hidden, 2)
    self.activation = torch.nn.ReLU()

  def forward(self, concepts, intervention_layer1 = None, intervention_layer2 = None, intervention_layer3 = None):
    x = self.layer1(concepts)
    l1 = self.activation(x)

    if intervention_layer1 is not None:
      l1 = intervention_layer1
    x = self.layer2(l1)
    l2 = self.activation(x)

    if intervention_layer2 is not None:
      l2 = intervention_layer2
    x = self.layer3(l2)
    l3 = self.activation(x)

    if intervention_layer3 is not None:
      l3 = intervention_layer3
    x = self.output(l3)
    sum_rhs = self.classifier(l3)
    sum_lhs = self.classifier2(l3)
    return x, sum_lhs, sum_rhs, l1, l2, l3


class CUSTOM_OPERATION(torch.nn.Module):
  def __init__(self, args: Namespace):
    super(CUSTOM_OPERATION, self).__init__()
    self.model_type = 'in-between'
    self.args = args
    self.encoder = ENCODER_MODULE()
    self.summation = MLP_MODULE(num_images=args.num_images)
    
  def forward(self, x,
              intervention_layer1 = None, intervention_layer2 = None, intervention_layer3 = None
              ):
    img1,img2,img3,img4 = torch.split(x, x.size(-1) // self.args.num_images, dim=-1)
    concept1 = self.encoder(img1)
    concept2 = self.encoder(img2)
    concept3 = self.encoder(img3)
    concept4 = self.encoder(img4)

    concepts = torch.cat([concept1,concept2,concept3,concept4],dim=-1).squeeze(1)
    out, sum_lhs, sum_rhs, l1,l2,l3 = self.summation(concepts, intervention_layer1, intervention_layer2, intervention_layer3)
    
    return out, sum_lhs, sum_rhs, concept1, concept2, concept3, concept4, l1, l2, l3
  

class SUMMATION_MODULE(torch.nn.Module):
  def __init__(self, c_dim = 10, hidden = 256, num_images = 2):
    super(SUMMATION_MODULE, self).__init__()
    self.layer1 = torch.nn.Linear(c_dim*num_images, hidden)
    self.layer2 = torch.nn.Linear(hidden, hidden)
    self.layer3 = torch.nn.Linear(hidden, hidden)
    self.classifier = torch.nn.Linear(hidden, c_dim*2-1)
    self.activation = torch.nn.ReLU()

  def forward(self, concepts, intervention_layer1 = None, intervention_layer2 = None, intervention_layer3 = None):
    x = self.layer1(concepts)
    l1 = self.activation(x)

    if intervention_layer1 is not None:
      #print('intervention_layer1')
      l1 = intervention_layer1
    x = self.layer2(l1)
    l2 = self.activation(x)

    if intervention_layer2 is not None:
      #print('intervention_layer2')
      l2 = intervention_layer2
    x = self.layer3(l2)
    l3 = self.activation(x)

    if intervention_layer3 is not None:
      #print('intervention_layer3')
      l3 = intervention_layer3
    
    sum = self.classifier(l3)
    
    return sum, l1, l2, l3
  
class ADDITION_SPLIT(torch.nn.Module):
  def __init__(self, args, c_dim = 10, hidden = 256, num_images = 2):
    super(ADDITION_SPLIT, self).__init__()
    self.model_type = 'only_sum_split'
    self.args = args
    self.encoder = ENCODER_MODULE(c_dim = c_dim)
    self.sum = SUMMATION_MODULE(c_dim = c_dim, hidden = hidden, num_images = num_images)
    
  def forward(self, x,
              intervention_layer1 = None, intervention_layer2 = None, intervention_layer3 = None):
    img1, img2 = torch.split(x, x.size(-1) // 2, dim=-1)
    concept1 = self.encoder(img1)
    concept2 = self.encoder(img2)
    concepts = torch.cat([concept1,concept2],dim=-1).squeeze(1)
    if intervention_layer1 is not None:
      sum,l1,l2,l3 = self.sum(concepts, intervention_layer1 = intervention_layer1)
    elif intervention_layer2 is not None:
      sum,l1,l2,l3 = self.sum(concepts, intervention_layer2 = intervention_layer2)
    elif intervention_layer3 is not None:
      sum,l1,l2,l3 = self.sum(concepts, intervention_layer3 = intervention_layer3)
    else:
      sum,l1,l2,l3 = self.sum(concepts)
    return sum, concept1, concept2, l1, l2, l3

class ADDITION_JOINT(torch.nn.Module):
  def __init__(self, args: Namespace, conv_channels = 4, c_dim = 20):
    super(ADDITION_JOINT, self).__init__()
    self.model_type = 'only_sum_joint'
    self.args = args
    self.conv_channels = conv_channels
    self.encoder1 = torch.nn.Sequential(
      torch.nn.Conv2d(1, conv_channels, kernel_size=4, stride=2, padding=(3,5)), # 16,32
      torch.nn.ReLU()
    )
    self.encoder2 = torch.nn.Sequential(
      torch.nn.Conv2d(conv_channels, conv_channels*2, kernel_size=4, stride=2, padding=1), # 8,16 *4
      torch.nn.ReLU()
    )
    self.encoder3 = torch.nn.Sequential(
      torch.nn.Conv2d(conv_channels*2, conv_channels*4, kernel_size=4, stride=2, padding=1), # 4,8 *16
      torch.nn.ReLU()
    )
    self.activation = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=0.4)
    self.dense_c = torch.nn.Linear(512, 128) # *4*8
    self.layer1 = torch.nn.Linear(128, 128)
    self.layer2 = torch.nn.Linear(128, 128)
    self.layer3 = torch.nn.Linear(128, 128)
    self.classifier = torch.nn.Linear(128, c_dim-1)
    
  def forward(self, x,
              intervention_encoder1 = None, intervention_encoder2 = None, intervention_encoder3 = None,
              intervention_layer1 = None, intervention_layer2 = None, intervention_layer3 = None
              ):
    batch_size = x.shape[0]
    x = self.encoder1(x)
    x = self.dropout(x)
    #print('x', x.shape)
    e1 = x.reshape(batch_size,-1)
    if intervention_encoder1 is not None:
      x = intervention_encoder1.reshape(batch_size,self.conv_channels,16,32)
    #print(x.shape)
    x = self.encoder2(x)
    x = self.dropout(x)
    e2 = x.reshape(batch_size,-1)
    if intervention_encoder2 is not None:
      x = intervention_encoder2.reshape(batch_size,self.conv_channels*2,8,16)
    #print(x.shape)
    x = self.encoder3(x)
    x = self.dropout(x)
    e3 = x.reshape(batch_size,-1)
    if intervention_encoder3 is not None:
      x = intervention_encoder3.reshape(batch_size,self.conv_channels*4,4,8) 
    #print(x.shape)
      
    
    x = self.dense_c(x.reshape(batch_size,-1))
  

    # MLP
    x = self.layer1(x)
    l1 = self.activation(x)
    if intervention_layer1 is not None:
      l1 = intervention_layer1

    x = self.layer2(l1)
    l2 = self.activation(x)
    if intervention_layer2 is not None:
      l2 = intervention_layer2
    
    x = self.layer3(l2)
    l3 = self.activation(x)
    if intervention_layer3 is not None:
      l3 = intervention_layer3
    
    x = self.classifier(l3)
    return x, e1, e2, e3, l1, l2, l3 

class DISEQ_MODULE(torch.nn.Module):
  def __init__(self, sum_dim = 19, hidden = 512, num_images = 2, out_dim = 2):
    super(DISEQ_MODULE, self).__init__()
    self.layer1 = torch.nn.Linear(sum_dim*num_images, hidden)
    self.layer2 = torch.nn.Linear(hidden, hidden)
    self.layer3 = torch.nn.Linear(hidden, hidden)
    self.classifier = torch.nn.Linear(hidden, out_dim)
    self.activation = torch.nn.ReLU()

  def forward(self, sums, intervention_layer1 = None, intervention_layer2 = None, intervention_layer3 = None):
    x = self.layer1(sums)
    l1 = self.activation(x)

    if intervention_layer1 is not None:
      l1 = intervention_layer1
    x = self.layer2(l1)
    l2 = self.activation(x)

    if intervention_layer2 is not None:
      l2 = intervention_layer2
    x = self.layer3(l2)
    l3 = self.activation(x)

    if intervention_layer3 is not None:
      l3 = intervention_layer3
    out = self.classifier(l3)
    
    return out, l1, l2, l3


class ALIGNED_CUSTOM_OPERATION(torch.nn.Module):
  def __init__(self, args: Namespace, conv_channels = 4, concepts_dim = 10, output_dim = 2):
    super(ALIGNED_CUSTOM_OPERATION, self).__init__()
    self.model_type = 'aligned'
    self.args = args
    self.conv_channels = conv_channels
    self.encoder = ENCODER_MODULE()
    self.summation = SUMMATION_MODULE(c_dim = concepts_dim)
    self.diseq = DISEQ_MODULE(sum_dim = (concepts_dim*2)-1, out_dim = output_dim)

  def forward(self, x,
              sum_layer1 = None, sum_layer2 = None, sum_layer3 = None,
              dis_layer1 = None, dis_layer2 = None, dis_layer3 = None
              ):
    batch_size = x.shape[0]
    img1,img2,img3,img4 = torch.split(x, x.size(-1) // self.args.num_images, dim=-1)
    concept1 = self.encoder(img1)
    concept2 = self.encoder(img2)
    concept3 = self.encoder(img3)
    concept4 = self.encoder(img4)

    if sum_layer1 is not None:
      ic1, ic2 = torch.split(sum_layer1, sum_layer1.size(-1) // 2, dim=-1)
      sum_1,s_l11,s_l12,s_l13 = self.summation(torch.cat([concept1,concept2],dim=-1).squeeze(1), intervention_layer1 = ic1)
      sum_2,s_l21,s_l22,s_l23 = self.summation(torch.cat([concept3,concept4],dim=-1).squeeze(1), intervention_layer1 = ic2)
    if sum_layer2 is not None:
      ic1, ic2 = torch.split(sum_layer2, sum_layer2.size(-1) // 2, dim=-1)
      sum_1,s_l11,s_l12,s_l13 = self.summation(torch.cat([concept1,concept2],dim=-1).squeeze(1), intervention_layer2 = ic1)
      sum_2,s_l21,s_l22,s_l23 = self.summation(torch.cat([concept3,concept4],dim=-1).squeeze(1), intervention_layer2 = ic2)
    if sum_layer3 is not None:
      ic1, ic2 = torch.split(sum_layer3, sum_layer3.size(-1) // 2, dim=-1)
      sum_1,s_l11,s_l12,s_l13 = self.summation(torch.cat([concept1,concept2],dim=-1).squeeze(1), intervention_layer3 = ic1)
      sum_2,s_l21,s_l22,s_l23 = self.summation(torch.cat([concept3,concept4],dim=-1).squeeze(1), intervention_layer3 = ic2)
    else:
      sum_1,s_l11,s_l12,s_l13 = self.summation(torch.cat([concept1,concept2],dim=-1).squeeze(1))
      sum_2,s_l21,s_l22,s_l23 = self.summation(torch.cat([concept3,concept4],dim=-1).squeeze(1))


    sums = torch.cat([sum_1,sum_2],dim=-1).squeeze(1)
    if dis_layer1 is not None:
      out,d_l1,d_l2,d_l3 = self.diseq(sums, intervention_layer1 = dis_layer1)
    if dis_layer2 is not None:
      out,d_l1,d_l2,d_l3 = self.diseq(sums, intervention_layer2 = dis_layer2)
    if dis_layer3 is not None:
      out,d_l1,d_l2,d_l3 = self.diseq(sums, intervention_layer3 = dis_layer3)
    else:
      out,d_l1,d_l2,d_l3 = self.diseq(sums)

    return out, concept1, concept2, concept3, concept4, sum_1, sum_2, d_l1, d_l2, d_l3, s_l11, s_l12, s_l13, s_l21, s_l22, s_l23
  



