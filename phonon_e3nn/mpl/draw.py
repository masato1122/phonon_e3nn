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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mytool.mpl.initialize import (set_matplot, set_axis, set_legend)

def draw(xdat, ydat, filename="fig.png", dpi=300,
        left=0.20, bottom=0.17, right=0.98, top=0.98):
    """
    xdat, ydat : array, float, shape=(ndat)
    """
    set_matplot()
    FIG_WIDTH = 3.3
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH))
    plt.subplots_adjust(left=left, bottom=bottom, right=right, 
        top=top, wspace=0, hspace=0)
    
    ax = plt.subplot()
    ax = set_axis(ax)
    
    plt.xlabel("")
    plt.ylabel("")
    
    lw = 1.0
    plt.plot(xdat, ydat, linestyle='None', c='#187FC4', 
            lw=lw, marker='o', markersize=2)
    
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print("Output:", filename)
    return 0

