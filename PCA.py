import pickle
from sklearn.decomposition import PCA
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

with open('/home/siddhaganju/recursion-ssl/all_features_FTSIMCLR.pkl', 'rb') as f:
  all_features = pickle.load(f)

with open('/home/siddhaganju/recursion-ssl/all_class_ids_FTSIMCLR.pkl', 'rb') as f:
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

for each in all_class_ids:
  # To include only the numeral portions
  new_each = int(each[6:])
  # print(new_each)
  new_class_ids.append(new_each)

# You can play with these values and see how the results change
n_components = 2
verbose = 1
perplexity = 40
n_iter = 1000
metric = 'euclidean'

time_start = time.time()
pca_results = PCA(n_components=n_components, tol=0.0).fit_transform(new_all_features)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

color_map = plt.cm.get_cmap('Spectral')
scatter_plot = plt.scatter(pca_results[:, 0],
                           pca_results[:, 1],
                           c=new_class_ids,
                           cmap=color_map)
plt.colorbar(scatter_plot)
plt.show()