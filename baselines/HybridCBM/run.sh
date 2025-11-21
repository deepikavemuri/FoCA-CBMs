python train.py --device 0 \
--cfg-options use_discri_loss=False \
--cfg-options use_ort_loss=False \
--cfg-options use_align_loss=False \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14'

python train.py --device 1 \
--cfg-options use_discri_loss=True \
--cfg-options use_ort_loss=False \
--cfg-options use_align_loss=False \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_LossDiscri'

python train.py --device 3 \
--cfg-options use_discri_loss=True \
--cfg-options use_ort_loss=True \
--cfg-options use_align_loss=False \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_Loss-Discri-Ort'

python train.py --device 4 \
--cfg-options use_discri_loss=True \
--cfg-options use_ort_loss=True \
--cfg-options use_align_loss=True \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_Loss-Discri-Ort-Align'

python train.py --device 0 \
--cfg-options use_discri_loss=True \
--cfg-options use_ort_loss=True \
--cfg-options num_static_concept=0 \
--cfg-options num_dynamic_concept=2000 \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_Dynamic_Loss-Discri-Ort-Align'

python train.py --device 2 \
--cfg-options use_discri_loss=False \
--cfg-options use_ort_loss=False \
--cfg-options use_normalize=True \
--cfg-options num_static_concept=2000 \
--cfg-options num_dynamic_concept=0 \
--cfg-options weight_init_method='zero' \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_Static_Norm'

python train_labo.py --device 2 \
--cfg-options use_discri_loss=False \
--cfg-options use_ort_loss=False \
--cfg-options num_static_concept=2000 \
--cfg-options num_dynamic_concept=0 \
--cfg-options use_normalize=True \
--cfg-options weight_init_method='label' \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_Static_Labo_Norm'

python train_labo.py --device 2 \
--cfg-options use_discri_loss=False \
--cfg-options use_ort_loss=False \
--cfg-options num_static_concept=2000 \
--cfg-options num_dynamic_concept=0 \
--cfg-options weight_init_method='cosine' \
--cfg-options exp_root='exp/HybridCBM/CUB/CUB_allshot_fac_ViT-L-14_Static_Labo_COS'