import os, sys
import numpy as np
import pandas as pd
import argparse
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mytool.mpl.initialize import (set_matplot, set_axis, set_legend)

def get_confidence_interval(values):
    from scipy.stats import bootstrap
    bst = bootstrap((values,), np.mean, confidence_level=0.9, random_state=42)
    low  = bst.confidence_interval.low
    high = bst.confidence_interval.high
    return low, high

def _read_file(filename):
    
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    
    target = None
    for key in ['valid_mae', 'mae_valid']:
        if key in df.columns:
            target = key
            break
    
    ibest = np.argmin(df[target].values)
    
    df_best = df.iloc[ibest].copy()
    
    seed = int(filename.split('/')[-2].replace('seed', ''))
    df_best['seed'] = int(seed)
    
    ###
    nall = 0
    for data_type in ['train', 'valid', 'test']:
        file_train = filename.replace('log.csv', f'idx_{data_type}.txt')
        lines = open(file_train).readlines()
        num_each = len(lines)
        df_best[f'num_{data_type}'] = num_each
        nall += num_each
    df_best['num_data'] = nall
    
    ###
    num_nominal = int(filename.split('/')[-3].split('_N')[-1])
    df_best['num_nominal'] = num_nominal
    
    return df_best

def read_log_csv(filename):
    df = _read_file(filename)
    out = {}
    out['num_nominal'] = df['num_nominal']
    for data_type in ['train', 'valid', 'test', 'data']:
        out[f'num_{data_type}'] = df[f'num_{data_type}']
    out['seed'] = df['seed']
    #
    out['train_mae'] = df['train_mae']
    out['train_mse'] = df['train_mse']
    out['train_custom'] = df['train_loss']
    #
    out['valid_mae'] = df['valid_mae']
    out['valid_mse'] = df['valid_mse']
    out['valid_custom'] = df['valid_loss']
    return out
    
def read_result_csv(filename):
    df = _read_file(filename)
    out = {}
    out['num_nominal'] = df['num_nominal']
    for data_type in ['train', 'valid', 'test', 'data']:
        out[f'num_{data_type}'] = df[f'num_{data_type}']
    
    out['seed'] = df['seed']
    #
    for data_type in ['train', 'valid', 'test']:
        out[f'{data_type}_mae'] =    df[f'mae_{data_type}']
        out[f'{data_type}_mse'] =    df[f'mse_{data_type}']
        out[f'{data_type}_custom'] = df['custom_error_'+data_type]
    
    return out

def _get_all_data(line="./out_N*"):
    
    all_dump = []
    
    dirs = glob.glob(line)
    for dd in dirs:
        
        dump = []
        # for ii in range(2):
        for ii in range(1):
            
            if ii == 0:
                line2 = dd + '/seed*/log.csv'
            else:
                line2 = dd + '/seed*/result.csv'
            
            fns = glob.glob(line2)
            
            for ff in fns:
                try:
                    out = read_result_csv(ff)
                except:
                    out = read_log_csv(ff)
                
                dump.append(pd.DataFrame(out, index=[0]))
        
        if len(dump) == 0:
            continue
        
        df_each = pd.concat(dump)
        
        each = {}
        for key in ['num_train', 'num_data', 'num_nominal']:
            each[key] = df_each[key].mean()
        
        for data_type in ['train', 'valid', 'test']:
            for error_type in ['loss', 'mse', 'mae']:
                key = data_type+'_'+error_type
                try:
                    each[key] = df_each[key].mean()
                    low, high = get_confidence_interval(df_each[key].values)
                    each[key+'_std'] = df_each[key].std()
                    each[key+'_low'] = each[key] - low
                    each[key+'_high'] = high - each[key]
                except:
                    pass
        
        all_dump.append(each)
    
    df_all = pd.DataFrame(all_dump)
    df_all.sort_values('num_train', inplace=True)
    print(df_all)
    return df_all

def make_frame(fontsize=7, fig_width=2.3, aspect=0.9, lw=0.5, ms=2.0):
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    ax = plt.subplot()
    return fig, ax

def main(options):
    
    df_all = _get_all_data(line='./out_N*')
    df_all = df_all.sort_values('num_train')
    
    # from plot_scaling import plot_scaling
    fig, ax = make_frame()
    
    markers = ['o', '^', 's']
    cmap = plt.get_cmap("tab10")
    for i, data_type in enumerate(['train', 'valid', 'test']):
        try:
            ax.plot(df_all['num_train'].values, 
                    df_all[f'{data_type}_mae'].values, 
                    label=data_type, marker=markers[i], 
                    ms=2.0, lw=0.5, color=cmap(i))
        except:
            pass
    
    # print(df_all.columns)
    # plot_scaling(
    #     ax, df_all,
    #     xcol='num_train',
    #     ycol='valid_mae',
    #     ylabel='MAE',
    # )
    
    set_axis(ax, xscale='log', yscale='log')
    set_legend(ax, fs=6, alpha=0.5)
    figname = 'fig_best_data.png'
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    print(f" Output {figname}")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters')

    parser.add_argument('-f', '--filename', dest='filename', type=str,
                        default="", help="input file name")

    parser.add_argument('-o', '--outfile', dest='outfile', type=str,
                        default=None, help="output file name")

    args = parser.parse_args()

    main(args)
