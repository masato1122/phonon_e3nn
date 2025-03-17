import numpy as np

def format_number(value, precision=2):
    format_str = f"{{:.{precision}f}}"
    return format_str.format(value)

def modify_ticklabels_log(
    ax, vmin, vmax, label_positions=None, which='x', minor=True,
    precision=2):
    """ Modify the axis label to be in log scale """
    if vmin < 0 or vmax < 0:
        print("Error: vmin or vmax is negative. Return None.")
        return None
    
    n0 = int(np.log10(vmin)) - 1
    n1 = int(np.log10(vmax)) + 1
    
    if minor:
        alist = [i for i in range(2, 10)]
    else:
        alist = [1]
    
    if label_positions is not None:
        label_positions = np.array(label_positions)
    
    minor_ticks = []
    minor_ticklabels = []
    for n in range(n0, n1+1):
        for a in alist:
            e = a * 10**n
            if e < vmin or vmax < e:
                continue
            minor_ticks.append(e)
            if label_positions is not None:
                if np.min(abs(label_positions - e)) < 10**n * 0.1:
                    line = format_number(e, precision)
                    minor_ticklabels.append(line)
                else:
                    minor_ticklabels.append('')
            else:
                if a in [1, 2, 5]:
                    line = format_number(e, precision)
                    minor_ticklabels.append(line)
                else:
                    minor_ticklabels.append('')
    
    # print()
    # print(minor)
    # print('ticks:', minor_ticks)
    # print('label:', minor_ticklabels)
    
    if which == 'x':
        ax.set_xticks(minor_ticks, minor=minor)
        ax.set_xticklabels(minor_ticklabels, minor=minor)
    elif which == 'y':
        ax.set_yticks(minor_ticks, minor=minor)
        ax.set_yticklabels(minor_ticklabels, minor=minor)

def scaling_function(x, a, d):
    return (10**d / x)**a

def get_scaling_law(xdat, ydat, function=scaling_function, p0=[0.1, -10.0]):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(
        function, xdat, ydat,
        maxfev=10**5,
        p0=p0,
        bounds=([0, -100], [20, 100]))
    return popt

def write_scaling_formula(ax, popt, pos=(0.05, 0.05), fontsize=7, ha='left', va='bottom'):
    line = "${\\rm MAE = (10^{%.2f} / N_{train})^{%.3f}}$" % tuple(popt[::-1])
    ax.text(pos[0], pos[1], line, fontsize=fontsize,
            transform=ax.transAxes, ha=ha, va=va,
            bbox=dict(facecolor='white', alpha=0.8, 
                      edgecolor='none', pad=1.5))

def plot_scaling_law(
    ax, xdat, ydat, color='grey', lw=3, alpha=0.5,
    params_formula={'pos': (0.05, 0.05), 'fontsize': 7, 'ha': 'left', 'va': 'bottom'},
    show_line=True, p0=None,
    ):
    """ Plot scaling law : y = (10^d / N_train)^a """
    popt = get_scaling_law(xdat, ydat, p0=p0)
    xfit = np.logspace(np.log10(xdat[0]), np.log10(xdat[-1]), 51)
    yfit = scaling_function(xfit, *popt)
    if show_line:
        ax.plot(xfit, yfit, linestyle='-', lw=lw, color=color, alpha=alpha)
    write_scaling_formula(ax, popt, **params_formula)
    return popt
