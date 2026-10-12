#!/bin/zsh
# Second queue: re-runs the 3 Diagonal seeds (stalled in queue 1) plus the 12 reviewer-requested
# experiments that queue 1 never launched. Same as run_queue.sh except:
#   -S 2048  raises BSD xargs' default 255-byte limit on the command line after {} is
#            substituted. The long "--model generic ..." lines exceeded it, so queue 1's xargs
#            refused to launch them ("command line cannot be assembled, too long").
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
P=${PYTHON:-python}   # set PYTHON to your env's interpreter (the original runs used a conda env named pvae)
mkdir -p logs
echo "queue2 start $(date)" >> logs/queue.log
cat queue2.txt | xargs -S 2048 -P 8 -I{} sh -c "$P train_one.py {} >> logs/train_stdout.log 2>&1; rc=\$?; echo \"\$(date) finished: {} (exit \$rc)\" >> logs/queue.log"
echo "queue2 end $(date)" >> logs/queue.log
