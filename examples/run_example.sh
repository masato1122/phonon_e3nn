#!/bin/sh

num_data=100
seed=1
outdir=./out_N${num_data}/seed${seed}

### for rough calculation
num_epochs=10; num_epochs_limit=10
### for accurate calculation
#num_epochs=1000; num_epochs_limit=1000

python ../tools/run_prediction.py \
    --file_data ../DATA/data_all.csv \
    --num_data  $num_data \
    --outdir    $outdir \
    --target    kcumu_norm_mfp \
    --seed      $seed \
    --r_max     4.3 \
    --num_epochs       $num_epochs \
    --num_epochs_limit $num_epochs_limit \
    --patience         50 \
    --batch_size   4 \
    --lr           0.002 \
    --lr_min       0.0005 \
    --weight_decay 0.03 \
    --gamma        0.95 

