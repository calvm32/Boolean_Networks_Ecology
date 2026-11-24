#!/bin/bash
#SBATCH --job-name=invasion
#SBATCH --output=results_and_data/20261120_121856_invasion_delta0.01_lat2SMW/logs/job.out
#SBATCH --error=results_and_data/20261120_121856_invasion_delta0.01_lat2SMW/logs/job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G

# Setup compute node environment
module purge
# MINIMAL CHANGE: Ensured openmpi is loaded on the compute node
module load compiler/gcc/11 openmpi/4.1 python/3.10

# Activate virtual environment
source "/home/velcsov/cheldt/envs/bn_ecology_env/bin/activate"

# Add top-level project directory to Python search path
export PYTHONPATH="/home/velcsov/cheldt/BN_ecology${PYTHONPATH:+:${PYTHONPATH}}"

# Export variables for Python multiprocessing (keeping variables for backwards compatibility)
export SIM_OUTPUT_DIR="results_and_data/20261120_121856_invasion_delta0.01_lat2SMW"
export SIM_NUM_CORES="1"
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

echo "==================================================="
echo "Starting Execution: invasion"
echo "==================================================="

mpirun python3 "/home/velcsov/cheldt/BN_ecology/simulate/compare_regimes/invasion.py"
