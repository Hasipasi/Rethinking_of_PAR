source /net/cremi/bbodis/Bureau/espaces/travail/VENV/bin/activate
echo "VENV activated"

# BASELINE ________________________________________________________________________________________________________________________________
# CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/rap_zs.yaml


# INCREMENTAL LEARNING ____________________________________________________________________________________________________________________
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_3.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_3.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_3.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_3.pkl

# CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/age_0.pkl
# CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/age_1.pkl

# CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/female_0.pkl

# VARIANCE CHECK 4000 ________________________________________________________________________________________________________________________
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_2.pkl  
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_3.pkl  

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_4.pkl  
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_5.pkl  
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_6.pkl  
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/bald_4000_7.pkl    

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_3.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_4.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_5.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_6.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/jacket_4000_7.pkl


CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_3.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_4.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_5.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_6.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_4000_7.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_3.pkl

CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_4.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_5.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_6.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/skirt_4000_7.pkl



# INCREMENTAL LEARNING WITH MULTIPLE ATTRIBUTES ______________________________________________________________________________________________
# CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_jacket_bald_skirt_0.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_jacket_bald_skirt_1.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_jacket_bald_skirt_2.pkl
CUDA_VISIBLE_DEVICES=0 python my_train.py --cfg ./configs/pedes_baseline/rap_zs.yaml --pkl ./data/RAP2/shirt_jacket_bald_skirt_3.pkl