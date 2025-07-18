#!/bin/bash
durations=(1 3 7 14)
simulation_names=(
    "single_basin_saigon"
    "single_basin_mekong"
    "single_basin_tonle_sap"
    "cascading_hazards_0d"
    "cascading_hazards_1d"
    "cascading_hazards_2d"
    "cascading_hazards_3d"
    "cascading_hazards_4d"
    "cascading_hazards_5d"
    "cascading_hazards_6d"
    "cascading_hazards_7d"
)
for duration in "${durations[@]}"; do
  for simulation_name in "${simulation_names[@]}"; do
  job_name="disr_khm${duration}_${simulation_name}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --error=logs/Cambodia/${job_name}_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
echo "Job started: \$(date)"
module purge
source /projects/disruptsc/miniforge3/bin/activate /projects/disruptsc/miniforge3/envs/dsc
python src/disruptsc/main.py Cambodia --cache_isolation --duration ${duration} --simulation_name ${simulation_name}
echo "Job ended: \$(date)"
EOF
    sleep 5
  done
done