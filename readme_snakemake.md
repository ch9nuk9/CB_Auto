# snakemake
my first snakemake repository

#To run:
Copy following files into your directory containing /data/ with the recordings inside data.
Snakemake
config.yaml
cluster_config.yaml
RUNME_cluster.sh

You will need to modify the config.yaml and the RUNME_cluster.sh files.

Then modify RUNME_cluster.sh so that it contains the number of files, in this case 5 in the variable NUM_JOBS_TO_SUBMIT:
```bash
#!/bin/bash

# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB

OPT="sbatch -p {cluster.partition} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output}"
NUM_JOBS_TO_SUBMIT=5

echo "Submitting $NUM_JOBS_TO_SUBMIT Jobs. Make sure that these are the number of files in the config.yaml file otherwise expect errors."

# Needs writable cache
# As of 8/2022 your home folder at /home/user should be writable from the cluster, but this may be temporary
# export HOME="/scratch/neurobiology/zimmer/YOUR/USER"

snakemake --configfile config.yaml --latency-wait 60 --use-conda --cluster "$OPT" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT
```

# Open a tmux session
```bash
tmux new -s bh
```
A new tmux session will open

Cd to your project folder
```
cd folder
bash RUNME_cluster.sh
```

## Resources:
This tutorial was helpful for me:
https://www.youtube.com/watch?v=r9PWnEmz_tc
