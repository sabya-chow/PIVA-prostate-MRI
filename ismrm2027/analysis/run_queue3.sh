#!/bin/zsh
# Third queue (round-2 reviews): per-voxel noise-SD likelihood x 3 seeds (Diag, Chol) plus seeds 1-2
# of the SD-50 runs. Same settings as run_queue2.sh (incl. the notes below), but 10 jobs at once.
#   -S 2048  raises BSD xargs' default 255-byte limit on the command line after {} is
#            substituted. The long "--model generic ..." lines exceeded it, so queue 1's xargs
#            refused to launch them ("command line cannot be assembled, too long").
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
P=${PYTHON:-python}   # set PYTHON to your env's interpreter (the original runs used a conda env named pvae)
mkdir -p logs
echo "queue3 start $(date)" >> logs/queue.log
cat queue3.txt | xargs -S 2048 -P 10 -I{} sh -c "$P train_one.py {} >> logs/train_stdout.log 2>&1; rc=\$?; echo \"\$(date) finished: {} (exit \$rc)\" >> logs/queue.log"
echo "queue3 end $(date)" >> logs/queue.log
