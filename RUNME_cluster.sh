#!/bin/bash
# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB
OPT="sbatch -p {cluster.partition} -t {cluster.time} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output}"
#NUM_JOBS_TO_SUBMIT=2
# Untested : Modify if necessary
NUM_JOBS_TO_SUBMIT=$(find $PWD -mindepth 3 -maxdepth 3 -type d -wholename "*/*/*w*/*Ch0" | wc -l)
echo "Submitting $NUM_JOBS_TO_SUBMIT Jobs. Make sure that these are the number of datasets otherwise expect errors."
# Needs writable cache
# As of 8/2022 your home folder at /home/user should be writable from the cluster, but this may be temporary
# export HOME="/scratch/neurobiology/zimmer/YOUR/USER"
snakemake --configfile config.yaml --latency-wait 240 --use-conda --cluster "$OPT" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT --rerun-incomplete
#snakemake --configfile config.yaml --latency-wait 240 --use-conda --cores 1