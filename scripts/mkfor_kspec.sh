#!/bin/sh
WORKDIR=$(cd $(dirname $0); pwd)

target=kspec_norm_freq; label="sp"
#target=kcumu_norm_mfp; label="cu"

nprocs=1

file_data=../../1_get/data_all4.csv

num_epochs=1000
nloop=1
num_epochs_limit=1000
patience=50

###################
id=18
#
batch=32
lr=0.001
weight_decay=0.03
gamma=0.93
###################

for seed in `seq 1 20`; do
for num_data in -1 3000 1000 300 100; do
#for num_data in -1 3000; do

    #if [ $num_data -lt 301 ]; then
    #    batch=32; lr=0.05
    #elif [ $num_data == "1000" ]; then
    #    batch=32; lr=0.03
    #elif [ $num_data == "3000" ]; then
    #    batch=32; lr=0.02
    #fi

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
    
    #lr=\`echo "0.005 - 0.0001 * \$i" | bc -l\`
    #
    #if [ \$lr -lt 0.0001 ]; then
    #    lr=0.002
    #fi
    #echo "lr = \${lr}"
    
    python ../run_prediction.py \\
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
        --lr           $lr \\
        --weight_decay $weight_decay \\
        --gamma        $gamma 
    
    
done
EOF
qsub $OFILE; sleep 0.1
#exit
done
done

