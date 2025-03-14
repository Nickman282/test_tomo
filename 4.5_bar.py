import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from common import load_params, MSE, sq_PSNR, SSIM, SSIM_2
from data_processor import Processor
from parametric_prior import HybridGibbsSampler
from matplotlib.ticker import FormatStrFormatter


PSNR_no_noise = [30.78, 26.68, 23.33, 27.42]
SSIM_no_noise = [0.901, 0.755, 0.495, 0.790]

PSNR_40 = [25.75, 21.86, 23.55, 26.52]
SSIM_40 = [0.794, 0.327, 0.447, 0.738]

PSNR_100 = [17.07, 15.20, 20.79, 22.79]
SSIM_100 = [0.348, 0.276, 0.158, 0.667]

cats = ['UGLA', 'RTO', 'NUTS', 'DIP']
fig, ax = plt.subplots()

y_pos = np.arange(len(cats))

color = 'thistle'
ax.barh((y_pos+0.125), PSNR_100, align='center', color=color, height=0.25, edgecolor='k')
ax.set_yticks(y_pos, labels=cats)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('PSNR [dB]', color=color)
ax.set_title('100 max. angle, no noise')
ax.grid()

ax2 = ax.twiny()  # instantiate a second Axes that shares the same x-axis

color = 'indigo'
ax2.set_xlabel('SSIM', color=color)  # we already handled the x-label with ax1
ax2.set_xlim(0, 1)
ax2.barh((y_pos-0.125), SSIM_100, align='center', color=color, height=0.25, edgecolor='k')
ax2.grid()

ax.set_xticks(np.linspace(0, ax.get_xbound()[1], 5))
ax2.set_xticks(np.linspace(0, ax2.get_xbound()[1], 5))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # ax2 is the RHS y-axis

fig.tight_layout()  # otherwise the right y-label is slightly clipped

'''
color = 'tab:red'
ax1.set_xlabel('PSNR (dB)')
ax1.set_ylabel('Algorithm', color=color)
ax1.bar(PSNR_no_noise, cats, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twiny()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
'''
plt.show()
