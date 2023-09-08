#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 14:43:25 2022

@author: kurtb
"""

import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
#%%
xs, _ = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 10))
plt.imshow(xs, cmap=cc.cm.colorwheel);  # use tab completion to choose

# #%%
# from PIL import Image
# import colorsys
# import math
#  #%%
# if __name__ == "__main__":
 
#     im = Image.new("RGB", (300,300))
#     radius = min(im.size)/2.0
#     cx, cy = im.size[0]/2, im.size[1]/2
#     pix = im.load()
 
#     for x in range(im.width):
#         for y in range(im.height):
#             rx = x - cx
#             ry = y - cy
#             s = (rx ** 2.0 + ry ** 2.0) ** 0.5 / radius
#             if s <= 1.0:
#                 h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
#                 rgb = colorsys.hsv_to_rgb(h, s, 1.0)
#                 pix[x,y] = tuple([int(round(c*255.0)) for c in rgb])
 
#     im.show()
#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl
plt.close('all')
# for cN in [k for k in cc.cm.keys() if ('cycl' in k) and (not '_r' in k)]:
fig = plt.figure()
# cmap = [cc.cm.colorwheel, # good but low sat
#         cc.cm.cyclic_bgrmb_35_70_c75, # ok 
#         cc.cm.cyclic_mrybm_35_75_c68,
#         ][-1]
# cmap = cc.cm[cN]
cmap = cc.cm.cyclic_ymcgy_60_90_c67_s25 # nice, but c, b might be better
# cmap = cc.cm.cyclic_rygcbmr_50_90_c64 # the vetical and lower are not aligned at primary colors
# cmap = cc.cm.cyclic_mybm_20_100_c48 # ok, but rg
cmap = cc.cm.colorwheel # good for colorblind (https://nicoguaro.github.io/posts/cyclic_colormaps/)
# twilight is also very good, but I worry that the greys might blend in with the sulci-gyri bakcground

display_axes = fig.add_axes([0.1,0.1,0.8,0.8], projection='polar')
display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                  ## multiply the values such that 1 become 2*pi
                                  ## this field is supposed to take values 1 or -1 only!!

norm = mpl.colors.Normalize(0.0, 2*np.pi)

# Plot the colorbar onto the polar axis
# note - use orientation horizontal so that the gradient goes around
# the wheel rather than centre out
quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(display_axes, cmap=cm.get_cmap(cmap,quant_steps),
                                   norm=norm,
                                   orientation='horizontal')

# aesthetics - get rid of border and axis labels                                   
cb.outline.set_visible(False)                                 
display_axes.set_axis_off()
# plt.title(cN)
plt.show()