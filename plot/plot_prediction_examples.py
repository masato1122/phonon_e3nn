# -*- coding: utf-8 -*-
# 
# title
# 
# Description :
# This script ...
# 
# Author      : M. Ohnishi
# Created on  : February 16, 2025
# 
import sys, os
import numpy as np
import pandas as pd
import argparse

#sys.path.append('/home/ohnishi/work/apdb/All_data/dos_spectral/4_prediction/phonon_e3nn')
from phonon_e3nn.utils.utils_data import set_data, plot_predictions_mod

def main(options):
    
    df_raw = pd.read_csv(options.filename)
    df, _ = set_data(df_raw, target=options.target)
    df = df.reset_index(drop=True)
    
    print(df)
    
    idx_test = df.index[df['kind'] == 'test'].tolist()
    
    if options.target.endswith('freq'):
        xcol = 'phfreq'
    elif options.target.endswith('mfp'):
        xcol = 'log_mfp'
    elif options.target.endswith('kspec_mod'):
        xcol = 'phfreq'
    else:
        print(options.target)
        print("Error: target name")
        sys.exit()
    
    errors = []
    for i in range(len(df)):
        vorig = np.asarray(df[options.target].values[i])
        vpred = np.asarray(df[options.target+'_pred'].values[i])
        mae = np.mean(np.abs(vorig - vpred))
        errors.append(mae)
    
    df['mae'] = errors
    
    #print(df.columns)
    plot_predictions_mod(
        df, idx_test, 
        loss_type='mae',
        title="Testing", 
        xcol=xcol, 
        ymin_left=options.ymin_left,
        ymax_left=options.ymax_left,
        target=options.target, 
        figname=options.figname,
        fig_width=options.fig_width,
        aspect=options.aspect,
        dpi=options.dpi,
        ncols=options.ncols,
        color_true='black'
        )
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters')

    parser.add_argument('-f', '--filename', dest='filename', type=str,
                        default="kcumu_norm_mfp/out_N-1/seed1/data_pred.csv", 
                        help="input file name")
    
    parser.add_argument('--target', dest='target', type=str,
                        default="kcumu_norm_mfp", help="target name [kcumu_norm_mfp]")
    
    parser.add_argument('--figname', dest='figname', type=str,
                        default='fig_prediction_examples.png', help="figure name")
    parser.add_argument('--dpi', dest='dpi', type=int,
                        default=600, help="dpi [600]")
    parser.add_argument('--fig_width', dest='fig_width', type=float,
                        default=6.0, help="fig_width [6.0]")
    parser.add_argument('--aspect', dest='aspect', type=float,
                        default=0.35, help="aspect [0.4]")
    
    parser.add_argument('--ymin_left', dest='ymin_left', type=float,
                        default=None, help="ymin_left [None]")
    parser.add_argument('--ymax_left', dest='ymax_left', type=float,
                        default=None, help="ymax_left [None]")

    parser.add_argument('--ncols', dest='ncols', type=int,
                        default=5, help="ncols [5]")

    args = parser.parse_args()

    main(args)
