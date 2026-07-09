#!/bin/bash
ssh star2 "sbatch" << 'EOF'
#!/bin/bash
#SBATCH --gres=gpu:1         # Number of GPUs (per node)
#SBATCH --mem=64G               # memory (per node)
#SBATCH --time=5-15:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=32         # Number of CPUs (per task)
#SBATCH --nodelist=cs-venus-16
#SBATCH --output=/localscratch/dya78/test.out
#SBATCH -J test

echo 'Start'

source ~/.zshrc
cd /localscratch/dya78/lightgen/TEXGen
conda activate texgen

echo "pwd: $(pwd)"
echo 'test'
echo 'Job End'

echo 'End'
EOF