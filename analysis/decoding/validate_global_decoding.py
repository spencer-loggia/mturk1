#!/usr/bin/env python
# coding: utf-8

# # A Notebook for Validating Global Convolutional Cross Decoding Model as in MTurk1 #
# 
# First we will test the simple case. In this case, there are two modalities, each decodale in 6 ROIs. Two rois in each set will be cross decodable, displaying the exact pattern shown in the same location from the other stimulus

# In[1]:


import importlib
import neurotools 
import torch 
import numpy as np
import val_dataloader
import decode_policy
val_dataloader = importlib.reload(val_dataloader)
decode_policy = importlib.reload(decode_policy)
neurotools = importlib.reload(neurotools)
SimpleValDataLoader = val_dataloader.SimpleValDataLoader
MapDecode2D = decode_policy.MapDecode2D
from matplotlib import pyplot as plt


# In[2]:


vdl = SimpleValDataLoader(spatial=32, epochs=2000)
vdl.plot_template_pattern()
# dark blue: a2a
# teal: a2b
# green: b2a
# yellow: b2b


# In[3]:


stim, target = vdl.generate_example(1, 0.3, "a", use_pattern=True)
plt.imshow(stim)
plt.show()


# In[4]:


stim, target = vdl.generate_example(1, 0.3, "b", use_pattern=True)
plt.imshow(stim)
plt.show()


# In[5]:
KERNEL = 2

vdl = SimpleValDataLoader(spatial=32, epochs=2000, warp=True)
ax_decoder = neurotools.decoding.SearchlightDecoder(KERNEL, spatial=(32, 32), pad=1, stride=1, lr=.01, reg=.1, device="cuda",
                                                 nonlinear=False, n_classes=3, reweight=True, channels=1)
print(ax_decoder.weights[-1].shape)

ax_decoder.fit(vdl[0])


bx_decoder = neurotools.decoding.SearchlightDecoder(KERNEL, spatial=(32, 32), pad=1, stride=1, lr=.01, reg=.1, device="cuda",
                                                   nonlinear=False, n_classes=3, reweight=True, channels=1)
bx_decoder.fit(vdl[1])

vdl = SimpleValDataLoader(spatial=32, epochs=1000)
maps = [[None, None],
       [None, None]]

maps[0][0] = ax_decoder.evaluate(vdl[0])[1]
maps[0][1] = ax_decoder.evaluate(vdl[1])[1]

vdl = SimpleValDataLoader(spatial=32, epochs=1000)
maps[1][0] = bx_decoder.evaluate(vdl[0])[1]
maps[1][1] = bx_decoder.evaluate(vdl[1])[1]

acc_maps = [[None, None],
       [None, None]]

acc_maps[0][0] = ax_decoder.evaluate(vdl[0])[0]
acc_maps[0][1] = ax_decoder.evaluate(vdl[1])[0]

print(acc_maps[0][0].mean())


