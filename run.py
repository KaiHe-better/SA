import numpy as np
import sys
from modules.train import main
import torch

# CUDA_LAUNCH_BLOCKING=1

if __name__ == '__main__':
    AE_F_res_list = []
    OE_F_res_list = []
    SC_F_res_list = []
    ABSA_F_res_list = []
    epoch_list = []
    for i in range(3):
        best_epoch, best_dic_F, opt = main(i)
        AE_F_res_list.append(best_dic_F["AE_F"])
        OE_F_res_list.append(best_dic_F["OE_F"])
        SC_F_res_list.append(best_dic_F["SC_F"])
        ABSA_F_res_list.append(best_dic_F["ABSA_F"])
        epoch_list.append(best_epoch)
        print("\n####################\n")

    with open("./results/output/"+str(sys.argv[1:])+"/total_res.txt", "w") as f:
        for index, (eopch, AE_F, OE_F, SC_F, ABSA_F)  in enumerate(zip(epoch_list, AE_F_res_list, OE_F_res_list, SC_F_res_list, ABSA_F_res_list)):
            f.write("index: {}, best epoch: {}".format(index, eopch)+"\n")
            f.write("AE_F: {:.2f}, OE_F: {:.2f}, SC_F: {:.2f}, ABSA_F: {:.2f}".format((AE_F), (OE_F), (SC_F), (ABSA_F))+"\n\n")

        f.write("\n\n")
        f.write("total res: \n")
        f.write("AE_F: {:.2f} ({:.2f}), OE_F: {:.2f} ({:.2f}), SC_F:{:.2f}, ({:.2f}), ABSA_F:{:.2f} ({:.2f})\n".format(np.mean(AE_F_res_list), np.std(AE_F_res_list),
                                                                                        np.mean(OE_F_res_list), np.std(OE_F_res_list),
                                                                                        np.mean(SC_F_res_list), np.std(SC_F_res_list),
                                                                                        np.mean(ABSA_F_res_list), np.std(ABSA_F_res_list) ))
    if opt.never_end:
        opt.ID = "temp"
        opt.epochs = 100000
        opt.total_epochs = 100000
        main(0, opt.ID)