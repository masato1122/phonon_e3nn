#
# Created by M. Ohnishi
# Created on February 06, 2025
# 
# MIT License
# 
# Copyright (c) 2024 Masato Ohnishi at The Institute of Statistical Mathematics
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import *
import matplotlib.gridspec as gridspec

def set_matplot(fontsize=9):
    lw_bor = 0.5
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["mathtext.fontset"] = 'dejavusans'
    plt.rcParams['axes.linewidth'] = lw_bor
    plt.rcParams['xtick.major.width'] = lw_bor
    plt.rcParams['xtick.minor.width'] = lw_bor
    plt.rcParams['ytick.major.width'] = lw_bor
    plt.rcParams['ytick.minor.width'] = lw_bor 

def set_spaces(plt,
        left=0.14, bottom=0.14, right=0.98, top=0.98, ratio=1.0,
        wspace=0., hspace=0.
        ):
    plt.subplots_adjust(
            left=left, bottom=bottom,
            right=right, top=top, wspace=wspace, hspace=hspace)

def set_axis(ax, 
        xformat=None, yformat=None,
        xscale="linear", yscale="linear", 
        xticks=None, mxticks=None, yticks=None, myticks=None,
        labelbottom=None, length=2.4, width=0.5):
    ax.tick_params(axis='both', which='major', 
            direction='in', length=length, width=width)
    ax.tick_params(axis='both', which='minor',
            direction='in', length=length*0.6, width=width)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    #--- for linear scale
    if xticks is not None:
        ax.xaxis.set_major_locator(tick.MultipleLocator(xticks))
    if mxticks is not None:
        interval = float(xticks) / float(mxticks)
        ax.xaxis.set_minor_locator(tick.MultipleLocator(interval))
    if yticks is not None:
        ax.yaxis.set_major_locator(tick.MultipleLocator(yticks))
    if myticks is not None:
        interval = float(yticks) / float(myticks)
        ax.yaxis.set_minor_locator(tick.MultipleLocator(interval))
    
    #--- for logscale
    if xformat is not None:
        xscale = xformat
    if yformat is not None:
        yscale = yformat
    if xscale.lower() == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(tick.LogLocator(base=10.0, numticks=15))
    if yscale.lower() == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(tick.LogLocator(base=10.0, numticks=15))
    return ax

def get_both_axis(erange, ylabel, ksym, klabels, x2label):
    gs = gridspec.GridSpec(1,3)
    ax1 = plt.subplot(gs[0,:2])
    ax2 = plt.subplot(gs[0,2])
    set_axis(ax1)
    set_axis(ax2)
    if ksym is not None:
        ax1.set_xticks(ksym)
    if klabels is not None:
        ax1.set_xticklabels(klabels)

    ax1.set_ylim(erange)
    ax2.set_ylim(erange)
    ax2.yaxis.set_major_formatter(NullFormatter())
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(x2label)
    return ax1, ax2

def set_legend(plt, ncol=1, fs=7, loc="best", loc2=None, handles=None,
        alpha=1.0, lw=0.2, length=1.0, labelspacing=0.3, borderpad=None,
        title=None, edgecolor='black', facecolor='white'):
    leg = plt.legend(
            loc=loc, ncol=ncol, fontsize=fs, fancybox=False, handles=handles, 
            facecolor=facecolor, edgecolor=edgecolor, handletextpad=0.4,
            handlelength=length, labelspacing=labelspacing,
            borderpad=borderpad, title=title, title_fontsize=fs)
    if loc2 is not None:
        leg.set_bbox_to_anchor([loc2[0], loc2[1]])
    leg.get_frame().set_alpha(alpha)
    leg.get_frame().set_linewidth(lw)	
    return leg

def set4bandos():
    FIG_WIDTH = 3.3
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.9))
    plt.subplots_adjust(
            left=0.14, bottom=0.14,
            right=0.98, top=0.98, wspace=0, hspace=0)
    return fig, plt

def set_axis_lim(ax, data, axis='x', alpha=0.05, scale='linear'):
    if scale == 'linear':
        vmin = np.min(data)
        vmax = np.max(data)
        x0 = vmin - alpha*(vmax - vmin)
        x1 = vmax + alpha*(vmax - vmin)
    elif scale == 'log':
        cmin = np.log10(np.min(data))
        cmax = np.log10(np.max(data))
        c0 = cmin - alpha*(cmax - cmin)
        c1 = cmax + alpha*(cmax - cmin)
        x0 = np.power(10, c0)
        x1 = np.power(10, c1)
    else:
        return None
    if axis == 'x':
        ax.set_xlim([x0, x1])
    else:
        ax.set_ylim([x0, x1])

def set_second_axis(ax):
    ax2 = ax.twinx()
    set_axis(ax)
    set_axis(ax2)
    ax.tick_params(labelright=False, right=False, which='both')
    ax2.tick_params(labelleft=False, left=False, which='both')
    return ax2

def set_axis_range(ax, values, which='x', scale='linear', margin=0.05):
    
    vmin = np.min(values)
    vmax = np.max(values)
    
    if scale == 'log':
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
    
    dv = vmax - vmin
    v0 = vmin - margin * dv
    v1 = vmax + margin * dv
    
    ##
    if scale == 'log':
        v0 = np.power(10, v0)
        v1 = np.power(10, v1)

    ##
    if which == 'x':
        ax.set_xlim([v0, v1])
    else:
        ax.set_ylim([v0, v1])
    

