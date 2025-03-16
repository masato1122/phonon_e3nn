
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

