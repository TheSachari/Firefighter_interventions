#!/bin/bash
#SBATCH --job-name=agent_run
#SBATCH --output=log/agent-%j.out
#SBATCH --error=log/agent-%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-type=END             
#SBATCH --mail-user=michael.corbeau@irit.fr    

export WANDB_API_KEY=$(< ~/.wandb_key)


container=/apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif
script=agent_run.py


model="dqn"   
years="10y"   
suffix="nsteps_40"

cmd="python3 -u ${script} \
  --model_name agent_${model}_${years}_${suffix} \
  --agent_model ${model} \
  --hyper_params hyper_params.json \
  --reward_weights rw_sent_disp.json \
  --dataset df_pc_fake_${years}.pkl \
  --start 1 \
  --end 530880 \
  --eps_start 1 \
  --constraint_factor_veh 3 \
  --constraint_factor_ff 1 \
  --save_metrics_as agent_metrics_${model}_${years}_${suffix} \
  --train"

srun singularity exec --nv ${container} bash -c "${cmd}"
