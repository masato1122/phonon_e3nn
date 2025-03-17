
# data pre-processing and visualization
import os
import sys
import os.path
import numpy as np
import pandas as pd
import argparse
import torch

#sys.path.append('/home/ohnishi/work/apdb/All_data/dos_spectral/4_prediction/phonon_e3nn')
from phonon_e3nn.prediction import run_simulation

## Clean data
def clean_data(df, tol1={'gap': 10, 'kappa': 500}, tol2={'kappa': 2000}):
    
    n0 = len(df)
    
    ## Remove too large gap and kappa
    df = df[~((df["max_gap"] >= tol1['gap']) & (df["kp"] >= tol1['kappa']))]
    df = df.reset_index(drop=True)
    
    ## Remove too small gap and kappa
    df = df[~(df["kp"] >= tol2['kappa'])]
    df = df.reset_index(drop=True)
    
    return df

def main(options):

    print("")
    print("")
    print(" START")
    print("")
    
    if options.nprocs is not None:
        torch.set_num_threads(options.nprocs)
    
    if os.path.exists(options.file_data) == False:
        print(" %s not found" % options.file_data)
        sys.exit()

    os.makedirs(options.outdir, exist_ok=True)
    
    ### "num_data" is changed to observe the scaling law
    if options.num_data == -1:
        num_data = None
    else:
        num_data = options.num_data        # None or integer
    
    ## Load data
    df_raw = pd.read_csv(options.file_data)
    df_raw = df_raw[(df_raw['fc2_error'] < 0.1) & (df_raw['fc3_error'] < 0.1)]
    
    if options.which_relax == 'both':
        pass
    elif options.which_relax == 'normal':
        df_raw = df_raw[df_raw['relax_type'] == 'normal']
    elif options.which_relax == 'strict':
        df_raw = df_raw[df_raw['relax_type'] == 'strict']
    else:
        print("Unknown relax_type")
        sys.exit()
    
    if len(df_raw) < 1000:
        print("Too small data size", len(df_raw))
        sys.exit()
    
    norig = len(df_raw)
    
    ## Clean data
    df_raw = clean_data(df_raw)
    
    ## Add log data
    if options.target == 'log_kp':
        df_raw['log_kp'] = np.log10(df_raw['kp'])
    elif options.target == 'log_kc':
        df_raw['log_kc'] = np.log10(df_raw['kc'])
    elif options.target == 'log_klat':
        df_raw['log_klat'] = np.log10(df_raw['klat'])
    
    ## Drop NaN
    df_raw = df_raw.dropna(subset=[options.target])
    navail = len(df_raw)
    
    print()
    print(f'Number of original data  : {norig}')
    print(f"Number of available data : {navail}")
    
    ## Sample data
    if num_data is not None:
        df_raw = df_raw.sample(n=num_data, random_state=options.seed)
        print(f"Number of sampled data   : {len(df_raw)}")
    
    ## Reset index
    df_raw = df_raw.reset_index(drop=True)
    
    
    ### alpha : weight for monotonicity penalty
    if 'cumu' in options.target.lower():
        mono_increase = True
        alpha = options.gradient_weight
    else:
        mono_increase = False
        alpha = 0.0
    
    ## Run simulation
    print()
    print('===============================')
    print('     Start simulation')
    print('===============================')
    print()
    run_simulation(
        df_raw,
        seed=options.seed,
        target=options.target,  # 'kspec_norm', 'kspec_mfp'
        outdir=options.outdir,
        r_max=options.r_max,
        valid_size=options.valid_size,
        test_size=options.test_size,
        batch_size=options.batch_size,
        num_epochs=options.num_epochs,
        num_epochs_limit=options.num_epochs_limit,
        patience=options.patience,
        plot_result=True,
        mono_increase=mono_increase,
        lr=options.lr,
        weight_decay=options.weight_decay,
        gamma=options.gamma,
        grad_weight=alpha,
        optimizer=options.optimizer.lower(),
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters')

    parser.add_argument('--nprocs', dest='nprocs', type=int,
                        default=1, help="nprocs [1]")

    parser.add_argument('--file_data', dest='file_data', type=str,
                        default="../1_get/data_all.csv", help="data file name")
    
    parser.add_argument('--outdir', dest='outdir', type=str,
                        default='./out', help="output directory [./out]")
    
    parser.add_argument('--target', dest='target', type=str,
                        default="kspec_norm", 
                        help="target (kspec_norm, kspec_mfp, phdos, ...) [kspec_norm]")
    
    parser.add_argument('--which_relax', dest='which_relax', type=str,
                        default="both", 
                        help="which_relax (both, normal, strict) [both]")
    
    parser.add_argument('--num_data', dest='num_data', type=int,
                        default=None, help="output directory [./out]")
    
    parser.add_argument('--seed', dest='seed', type=int,
                        default=12, help="seed [12]")

    parser.add_argument('--r_max', dest='r_max', type=float,
                        default=4.0, help="r_max [4.0]")
    
    parser.add_argument('--valid_size', dest='valid_size', type=float,
                        default=0.1, help="valid_size [0.1]")
    parser.add_argument('--test_size', dest='test_size', type=float,
                        default=0.1, help="test_size [0.1]")
    
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32, help="batch_size [32]")
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        default=3, help="num_epochs [3]")
    parser.add_argument('--num_epochs_limit', dest='num_epochs_limit', type=int,
                        default=None, help="max. limit of num_epochs [None]")
    parser.add_argument('--patience', dest='patience', type=int,
                        default=50, help="patience for loss increasing [50]")
    
    parser.add_argument('--gradient_weight', dest='gradient_weight', type=float,
                        default=0.0, 
                        help="weight for monotonicity penalty for kcumu [0.0]")
    
    ###
    ### Recommended:
    ### lr = 5.0 / N_all
    ### lr_min = 1.5 / N_all
    ### gamma = 0.95
    ###
    parser.add_argument('--lr', dest='lr', type=float,
                       default=0.001, 
                       help=(
                           "learning rate (orig: 0.005, general: 0.001, 0.0001) [0.001]. "
                           "Decrease if overfitting."))
    parser.add_argument('--lr_min', dest='lr_min', type=float,
                       default=0.0001, 
                       help="minimum learning rate [0.0001]]")
    
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=0.03, 
                        help=(
                            "weight decay (orig: 0.05, general: 0.01, 0.001) [0.03]. "
                            "Increase if overfitting."))
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.95, help="gamma [0.95]")
    
    parser.add_argument('--optimizer', dest='optimizer', type=str,
                       default='adam', 
                       help="optimizer (adam or adamw) [adam]]")
    
    args = parser.parse_args()
    
    ## Save parameters
    os.makedirs(args.outdir, exist_ok=True)
    file_params = os.path.join(args.outdir, 'params.txt')
    with open(file_params, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key:17s} : {value}\n')
    
    main(args)
    
