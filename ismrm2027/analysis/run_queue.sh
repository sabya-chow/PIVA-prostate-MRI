#!/bin/zsh
# Runs every line of queue.txt as a train_one.py job, 8 at a time, 1 thread each.
# (Benchmarked on this i9-9880H: 8 x 1-thread jobs give the highest total throughput.)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
P=${PYTHON:-python}   # set PYTHON to your env's interpreter (the original runs used a conda env named pvae)
mkdir -p logs
echo "queue start $(date)" >> logs/queue.log
cat queue.txt | xargs -P 8 -I{} sh -c "$P train_one.py {} >> logs/train_stdout.log 2>&1; echo \"\$(date) finished: {}\" >> logs/queue.log"
echo "queue end $(date)" >> logs/queue.log
