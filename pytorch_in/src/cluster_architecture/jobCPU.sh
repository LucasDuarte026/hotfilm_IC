#PBS -N NN_architecture
#PBS -l ncpus=1
#PBS -l walltime=48:00:00
#PBS -M lucassalesduarte026@gmail.com
#PBS -m abe

echo "Seu código que será gerado: $1"
echo "Há $2 camadas com $3 neurônios cada"

cd /home/lucasdu/algoritmo/cluster_architecture
source ~/cluster_torch/bin/activate
python3 ./model_mlp_v8.0_cluster.py $1 $2 $3 > $1.txt