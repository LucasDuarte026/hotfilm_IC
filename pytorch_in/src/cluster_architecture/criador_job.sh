#!/bin/bash

# nome do arquivo / num de camadas / num de neuronios

name=$1
layers=$2
neurons=$3

output_file="job_$name.sh"

cat > "$output_file" <<EOF

#PBS -N NN_$name
#PBS -l ncpus=1
#PBS -l walltime=48:00:00
#PBS -M lucassalesduarte026@gmail.com
#PBS -m abe

echo "Seu código que será gerado: $name"
echo "Há $layers camadas com $neurons neurônios cada"

cd /home/lucasdu/algoritmo/cluster_architecture
source ~/cluster_torch/bin/activate
python3 ./model_mlp_v8.0_cluster.py $name $layers $neurons > $name.txt
EOF

chmod +x "$output_file"
echo "Created bash file: $output_file"
