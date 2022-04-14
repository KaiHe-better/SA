srun -p NV100q  -w node23 nvidia-smi ; srun -p NV100q  -w node24 nvidia-smi
srun -p PV1003q  -w node14 nvidia-smi; srun -p PV1003q  -w node15 nvidia-smi; srun -p PV1003q  -w node16 nvidia-smi; srun -p PV1003q  -w node17 nvidia-smi;
srun -p PA40q  -w node01 nvidia-smi; srun -p PA40q  -w node04 nvidia-smi; srun -p PA40q  -w node13 nvidia-smi
srun -p RTX8Kq  -w node22 nvidia-smi;


# nohup python run.py --ID 0 --GPU 2 --if_weight_loss --my_min 3.5 --my_max 4 >/dev/null 2>&1 &  best epoch 104, best_f_absa 74.70 
# nohup python run.py --ID 1 --GPU 3 --if_weight_loss --my_min 3 --my_max 3.5 >/dev/null 2>&1 &  best epoch 144, best_f_absa 74.93 


# srun -p PV1003q -w node17 nohup  python run.py --ID N17 --GPU 2   >/dev/null 2>&1 & best epoch 39, best_f_absa 73.80 
# srun -p NV100q  -w node23 nohup  python run.py --ID N23 --GPU 3  --if_weight_loss --mix_loss_weight 0.3 --data_ratio 0.6 >/dev/null 2>&1 &  # best epoch 30, best_f_absa 73.83 
# srun -p NV100q  -w node24 nohup  python run.py --ID N24 --GPU 2  --if_weight_loss --mix_loss_weight 0.5 --data_ratio 0.7 >/dev/null 2>&1 &  # best epoch 22, best_f_absa 75.82 


# srun -p PV1003q -w node15 nohup  python run.py --ID N17 --GPU 1  --if_weight_loss --uncertain_th_S 0.4 --uncertain_th_E 0.8  >/dev/null 2>&1 &  # AE_F: 85.80 (0.50), OE_F: 85.47 (0.65), SC_F:79.95, (0.83), ABSA_F:73.74 (0.59)
# srun -p NV100q  -w node24 nohup  python run.py --ID N24 --GPU 2  --if_weight_loss --uncertain_th_S 0.4 --uncertain_th_E 0.9  >/dev/null 2>&1 &  # AE_F: 86.10 (0.47), OE_F: 84.26 (0.35), SC_F:79.73, (2.82), ABSA_F:74.35 (0.81)
# srun -p NV100q  -w node23 nohup  python run.py --ID N23 --GPU 3   >/dev/null 2>&1 & # AE_F: 87.25 (0.52), OE_F: 84.02 (0.02), SC_F:78.20, (0.53), ABSA_F:74.08 (0.40)


# srun -p PV1003q -w node15 nohup  python run.py --ID N15 --GPU 1  --if_weight_loss --uncertain_th_S 0.1 --uncertain_th_E 1.0  >/dev/null 2>&1 &  # total res: AE_F: 86.09 (0.15), OE_F: 84.72 (0.30), SC_F:78.47, (3.23), ABSA_F:73.45 (1.92)
# srun -p NV100q  -w node24 nohup  python run.py --ID N24 --GPU 2  --if_weight_loss --uncertain_th_S 0.3 --uncertain_th_E 1.0  >/dev/null 2>&1 &  # total res:  AE_F: 86.14 (0.39), OE_F: 84.57 (1.19), SC_F:79.27, (2.02), ABSA_F:73.80 (0.94)
# srun -p NV100q  -w node23 nohup  python run.py --ID N23 --GPU 3  --if_weight_loss --uncertain_th_S 0.2 --uncertain_th_E 1.0  >/dev/null 2>&1 &  # total res:  AE_F: 85.36 (0.23), OE_F: 84.44 (0.90), SC_F:77.49, (1.41), ABSA_F:72.31 (0.60)







srun -p NV100q  -w node24 nohup bash temp_run.sh >/dev/null 2>&1 & 


srun -p PA40q  -w node01 nohup python run.py --ID no_weig --GPU 1 --corpus res14 >/dev/null 2>&1 &


srun -p PV1003q  -w node14 nohup python run.py --ID all_lap14 --GPU 0 --corpus lap14 --data_ratio 1 --if_weight_loss --batch_size 8 --uncertain_th_S 0.3 --uncertain_th_E 0.8 --chosed_th 0.7 --add_other_corpus  >/dev/null 2>&1 &   