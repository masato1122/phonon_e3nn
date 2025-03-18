#!/bin/sh
WORKDIR=$(cd $(dirname $0); pwd)

target=kspec_norm_freq; label="sp"
#target=kcumu_norm_mfp; label="cu"

file_data=../../1_get/data_all5.csv

if [ $# -ge 1 ]; then
    id=$1
else
    echo -n "Input ID: "
    read id
fi

num_epochs=1000
nloop=1
num_epochs_limit=1000
patience=50

###################
##lr_init=0.005
##lr_min=0.0005
gamma=0.90
#
batch=16
weight_decay=0.01
optimizer=adamw
###################

for seed in `seq 11 20`; do
for num_data in -1 3000 1000 300 100; do

if [ $num_data -lt 1001 ]; then
    nprocs=4
else
    nprocs=1
fi

############################################
#lr_init=0.01
if [ $num_data == -1 ]; then
    lr_init=`echo "5.0 / 6000." | bc -l`
    lr_min=`echo "1.0 / 6000." | bc -l`
else
    lr_init=`echo "5.0 / $num_data" | bc -l`
    lr_min=`echo "1.0 / $num_data" | bc -l`
fi
############################################

jobname=j${label}${id}_${num_data}-${seed}

OFILE=a.sh
cat >$OFILE<<EOF
#!/bin/sh
#PBS -q workq
#PBS -l nodes=1:ppn=${nprocs}
#PBS -j oe
#PBS -N $jobname

export LANG=C
export OMP_NUM_THREADS=${nprocs}
cd \$PBS_O_WORKDIR
rm $jobname.o*

num_data=$num_data
seed=${seed}
outdir=./out_N\${num_data}/seed\${seed}

for i in \`seq 1 $nloop\`; do
    
    python ../phonon_e3nn/tools/run_prediction.py \\
        --file_data $file_data \\
        --num_data  $num_data \\
        --outdir    \$outdir \\
        --target    $target \\
        --seed      \$seed \\
        --r_max     4.3 \\
        --num_epochs       $num_epochs \\
        --num_epochs_limit $num_epochs_limit \\
        --patience         $patience \\
        --nprocs   ${nprocs} \\
        --batch_size   $batch \\
        --lr           $lr_init \\
        --lr_min       $lr_min \\
        --weight_decay $weight_decay \\
        --gamma        $gamma \\
        --optimizer    $optimizer
    
done
EOF
qsub $OFILE; sleep 0.1
done
done

