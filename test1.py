import os

bash_command  = "srun -p NV100q  -w node24 nvidia-smi"
os.system(bash_command)