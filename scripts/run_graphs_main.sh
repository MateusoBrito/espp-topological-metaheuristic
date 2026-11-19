TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/graphs_${TIMESTAMP}.log"

python3 -u -m src.main | tee -a "$LOG"