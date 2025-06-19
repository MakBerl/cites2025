srun -p all -G 1  python  ss_f1_score_diagnostics.py \
        --run-name=Dropout_01_run0001 \
        --src-dir=/home/cites/mloi0/cites2025/data/stage3/ \
        --dst-dir=./ss/ \
        --snapshot=./checkpoints/Dropout_01_run0001/best_model.pth \
        --batch-size=16
