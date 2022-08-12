import pickle 
import pandas as pd
import torch

with open('/home/siddhaganju/recursion-ssl/all_features_PTSIMCLR.pkl', 'rb') as f:
# with open('/home/siddhaganju/recursion-ssl/all_features_supervised_SIMCLR.pkl', 'rb') as f:
# with open('/home/siddhaganju/recursion-ssl/all_features_PTSIMCLR.pkl', 'rb') as f:
  all_features = pickle.load(f)

with open('/home/siddhaganju/recursion-ssl/all_class_ids_PTSIMCLR.pkl', 'rb') as f:
# with open('/home/siddhaganju/recursion-ssl/all_class_ids_supervised_SIMCLR.pkl', 'rb') as f:
# with open('/home/siddhaganju/recursion-ssl/all_class_ids_PTSIMCLR.pkl', 'rb') as f:
  all_class_ids = pickle.load(f)

with open('/home/siddhaganju/recursion-ssl/all_wells_PTSIMCLR.pkl', 'rb') as f:
# with open('/home/siddhaganju/recursion-ssl/all_class_ids_PTSIMCLR.pkl', 'rb') as f:
  all_wells = pickle.load(f)

new_all_features = []
new_class_ids = []

b = torch.tensor([10.12, 20.56, 30.00, 40.3, 50.4])
for each in all_features:
  if type(each) == type(b):
    numpy_each = each.detach().cpu().numpy()
    new_all_features.append(numpy_each)
  else:
    new_all_features.append(each)

for each in all_class_ids:
  # To include only the numeral portions
  # print(each)
  new_each = int(each[6:])
  # print(new_each)
  new_class_ids.append(new_each)

# for each in all_wells:
#   # To include only the numeral portions
#   # print(each)
#   # print(new_each)
#   all_wells.append(each)

# df = pd.DataFrame(new_all_features[:100])
# print(df.head(), len(df))
# df.to_csv(r'all_features_ncl_supervised_1channel.tsv', sep="\t", index=False, header=False)

# df = pd.DataFrame(new_class_ids[:100])
# print(df.head(), len(df))
# df.to_csv(r'all_class_ids_ncl_supervised_1channel.tsv', sep="\t", index=False, header=False)

df = pd.DataFrame(all_wells[:100])
print(df.head(), len(df))
df.to_csv(r'all_wells_ncl_PT_1channel.tsv', sep="\t", index=False, header=False)