import numpy as np
from musicnn_keras.extractor import extractor

import matplotlib.pylab as plt

file_name = './audio/joram-moments_of_clarity-08-solipsism-59-88.mp3'
taggram, tags = extractor(file_name, model='MTT_musicnn', extract_features=False)


in_length = 3 # seconds -- by default, the model takes inputs of 3 seconds with no overlap
# depict taggram
plt.rcParams["figure.figsize"] = (10,8)
fontsize=12
fig, ax = plt.subplots()
ax.imshow(taggram.numpy().T, interpolation=None, aspect="auto")

# title
ax.title.set_text('Taggram')
ax.title.set_fontsize(fontsize)

# x-axis title
ax.set_xlabel('(seconds)', fontsize=fontsize)

# y-axis
y_pos = np.arange(len(tags))
ax.set_yticks(y_pos)
ax.set_yticklabels(tags, fontsize=fontsize-1)

# x-axis
x_pos = np.arange(taggram.shape[0])
x_label = np.arange(in_length/2, in_length*taggram.shape[0], 3)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_label, fontsize=fontsize)

plt.show()