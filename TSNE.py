import pickle
from sklearn.manifold import TSNE
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import torch
import pandas as pd

with open('/home/siddhaganju/recursion-ssl/all_features_supervised_SIMCLR.pkl', 'rb') as f:
  all_features = pickle.load(f)

with open('/home/siddhaganju/recursion-ssl/all_class_ids_supervised_SIMCLR.pkl', 'rb') as f:
  all_class_ids = pickle.load(f)

# print(all_features)

new_all_features = []
new_class_ids = []

b = torch.tensor([10.12, 20.56, 30.00, 40.3, 50.4])
# print(len(all_features))
for each in all_features:
  if type(each) == type(b):
    numpy_each = each.detach().cpu().numpy()
    # print(type(numpy_each), type(each))
    new_all_features.append(numpy_each)
  else:
    new_all_features.append(each)

print(len(all_class_ids))
for each in all_class_ids:
  # To include only the numeral portions
  new_each = int(each[6:])
  # print(new_each)
  new_class_ids.append(new_each)

# You can play with these values and see how the results change
n_components = 2
verbose = 1
perplexity = 100
n_iter = 5000
metric = 'euclidean'

time_start = time.time()
tsne_results = TSNE(n_components=n_components, random_state=0,
                    verbose=verbose,
                    perplexity=perplexity,
                    learning_rate = 50,
                    method='exact',
                    n_iter=n_iter,
                    metric=metric).fit_transform(new_all_features[:120])

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

df = pd.DataFrame()
df["y"] = new_class_ids[:120]
df["comp-1"] = tsne_results[:,0]
df["comp-2"] = tsne_results[:,1]
#print(new_class_ids[:120])
color_map = plt.cm.get_cmap('coolwarm')
scatter_plot = plt.scatter(tsne_results[:, 0],
                           tsne_results[:, 1],
                           c=new_class_ids[:120])
                          #  cmap=color_map)

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("flare", 10),
#                 data=df).set(title="T-SNE projection") 
plt.colorbar(scatter_plot)
#plt.save("Supervised-SIMCLR.png")
plt.show()