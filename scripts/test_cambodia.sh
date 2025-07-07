#!/bin/bash
durations=(1 3 7 14 21 28)
simulation_names=(
    "simultaneous_disruptions"
    "single_basin_saigon"
    "single_basin_mekong"
    "single_basin_tonle_sap"
    "cascading_hazards_consecutive"
    "cascading_hazards_time_lag_short"
    "cascading_hazards_time_lag_long"
    "cascading_and_compound_hazards_consecutive"
    "cascading_and_compound_hazards_time_lag_short"
    "cascading_and_compound_hazards_time_lag_long"
    "cascading_and_compound_hazards_time_lag_very_long"
)
for duration in "${durations[@]}"; do
  for simulation_name in "${simulation_names[@]}"; do
  job_name="flood_khm${duration}_${simulation_name}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/Cambodia/${job_name}_%j.log
#SBATCH --error=logs/Cambodia/${job_name}_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
echo "Job started: \$(date)"
module purge
source /projects/disruptsc/miniforge3/bin/activate /projects/disruptsc/miniforge3/envs/dsc
python src/disruptsc/main.py Cambodia --duration ${duration} --simulation_name ${simulation_name}
echo "Job ended: \$(date)"
EOF
    sleep 5
  done
done