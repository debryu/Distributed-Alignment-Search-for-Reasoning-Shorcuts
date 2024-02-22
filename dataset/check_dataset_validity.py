from torch import Tensor, load
path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ML/data/custom_task/das_dataset_no_interventions.pt"
data = load(path)
train = data['das_dataset_no_interventions_train']['labels']
test = data['das_dataset_no_interventions_test']['labels']


def get_label(concepts_list:list) -> int:
    """Get label for the dataset."""
    assert len(concepts_list) == 4, "The number of concepts is not 4"
    # assuming 4 concepts
    c1 = concepts_list[0]
    c2 = concepts_list[1]
    c3 = concepts_list[2]
    c4 = concepts_list[3]
    res = 1 if c1+c2 > c3+c4 else 0
    return res

correct = 0
errors = 0
for sample in train:
    A,B,C,D,_ = sample
    inputt = [A,B,C,D]
    label = get_label(inputt)
    if label == 0:
      if (B==5 or D == 5) and (A!=5 and C !=5):
         correct += 1
      else:
         errors += 1
    elif label == 1:
      if (A==5 or C == 5) and (B!=5 and D !=5):
         correct += 1
      else:
         errors += 1
    else:
      errors += 1

print(f"Correct {correct} and {errors} errors in train!")



correct = 0
errors = 0
for sample in test:
    A,B,C,D,_ = sample
    inputt = [A,B,C,D]
    label = get_label(inputt)
    if label == 0:
      if (B==5 or D == 5) and (A!=5 and C !=5):
         correct += 1
      else:
         errors += 1
    elif label == 1:
      if (A==5 or C == 5) and (B!=5 and D !=5):
         correct += 1
      else:
         errors += 1
    else:
      errors += 1

print(f"Correct {correct} and {errors} errors in test!")