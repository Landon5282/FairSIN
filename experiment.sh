### vanilla
# gcn
python gnn.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 
python gnn.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01
python gnn.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01
# gin
python gnn.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 
python gnn.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 
python gnn.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 
# sage
python gnn.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 
python gnn.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 
python gnn.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=10 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.01 --e_lr=0.01 


### in-train gcn
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=0.5 --d='yes' --c_wd=0.001 --e_wd=0.001

python train_mlp.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25
# python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=32 --epoch=150  --delta=10 --d='yes'  # discriminator optimizer setting 

# python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=1 --d='yes'  
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1 
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0.1 
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=1 

###in-train gin
python in-train.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=80  --delta=1 --c_wd=0 --e_wd=0 --c_lr=0.1 --e_lr=0.01 --d='yes'  #--alpha=5
python in-train.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=80  --delta=1 --c_wd=0 --e_wd=0 --c_lr=0.01 --e_lr=0.01 --d='yes'  #--alpha=5

python train_mlp.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25 --d='yes'
python train_mlp.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25
python in-train.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=32 --epoch=100  --delta=10 --d='yes' 

python in-train.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --m_epoch=20 --delta=2  --c_lr=0.01 --e_lr=0.01
python in-train.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --m_epoch=20 --delta=1.25  --c_lr=0.01 --e_lr=0.01

### in-train sage
python in-train.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=100 --m_epoch=10 --dropout=0 --delta=1 --d='yes'
python train_mlp.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=100 --m_epoch=10 --dropout=0 --delta=1 --d='yes'

python in-train.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=10 --dropout=0.2 --delta=1 --c_lr=0.15 --e_lr=0.15 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=10 --dropout=0.2 --delta=0.5 --c_lr=0.15 --e_lr=0.15 --d='yes' --c_wd=0.001 --e_wd=0.001

python in-train.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=10 --dropout=0.8 --delta=0.1
python in-train.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=10 --dropout=0.8 --delta=0.1 --d='yes'



### ablation study
# german-gcn
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0 --d='no' 
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0 --d='yes' 
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=0.1 --d='no' 
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=0.1 --d='yes' 
# credit-gcn
python train_mlp.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0 --d='no' 
python train_mlp.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0 --d='yes' 
python train_mlp.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25
python train_mlp.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25 --d='yes'
# bail-gcn
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0 --d='no' 
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0 --d='yes' 
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0.1 --d='no' 
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0.1 --d='yes' 
# pokec_n-gcn
python in-train.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=0 --d='yes'
python in-train.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='no'
# pokec_z-gcn
python in-train.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=0 --d='yes'
python in-train.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='no'

# german-gin
python in-train.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100  --delta=0 --c_wd=0 --e_wd=0 --c_lr=0.01 --e_lr=0.01  --d='no'
python in-train.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100  --delta=0 --c_wd=0 --e_wd=0 --c_lr=0.01 --e_lr=0.01  --d='yes'
python in-train.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100  --delta=0.5 --c_wd=0 --e_wd=0 --c_lr=0.01 --e_lr=0.01  --d='no'
python in-train.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100  --delta=0.5 --c_wd=0 --e_wd=0 --c_lr=0.01 --e_lr=0.01 --d='yes' --alpha=2
#credit-gin
python train_mlp.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0 --d=='no'
python train_mlp.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0 --d='yes'
python train_mlp.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25 --d='no'
python train_mlp.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25 --d='yes'
#bail-gin
python in-train.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --m_epoch=20 --delta=0  --c_lr=0.01 --e_lr=0.01 --d='no'
python in-train.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --m_epoch=20 --delta=0  --c_lr=0.01 --e_lr=0.01  --d='yes'
python in-train.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --m_epoch=20 --delta=1.25  --c_lr=0.01 --e_lr=0.01
python in-train.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --m_epoch=20 --delta=1.25  --c_lr=0.01 --e_lr=0.01 --d='yes'
# pokec_n-gin
python in-train.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=0 --d='yes'
python in-train.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='no'
# pokec_z-gin
python in-train.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=0 --d='yes'
python in-train.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='no'

#german-sage
python in-train.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=100 --m_epoch=10 --dropout=0 --delta=0 --d='no'
python in-train.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=100 --m_epoch=10 --dropout=0 --delta=0 --d='yes'
python in-train.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=100 --m_epoch=10 --dropout=0 --delta=1 --d='no'
python in-train.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=100 --m_epoch=10 --dropout=0 --delta=1 --d='yes'
#credit-sage
python in-train.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=10 --dropout=0.2 --delta=0 --c_lr=0.15 --e_lr=0.15 --d='no' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=10 --dropout=0.2 --delta=0 --c_lr=0.15 --e_lr=0.15 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=10 --dropout=0.2 --delta=1.5 --c_lr=0.15 --e_lr=0.15 --d='no' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=10 --dropout=0.2 --delta=1.5 --c_lr=0.15 --e_lr=0.15 --d='yes' --c_wd=0.001 --e_wd=0.001
#bail-sage
python in-train.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=10 --dropout=0.8 --delta=0 --d='no'
python in-train.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=10 --dropout=0.8 --delta=0 --d='yes'
python in-train.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=10 --dropout=0.8 --delta=0.05 --d='no'
python in-train.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=10 --dropout=0.8 --delta=0.05 --d='yes'

# pokec_n-sage
python in-train.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=0 --d='yes'
python in-train.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='no'
# pokec_z-sage
python in-train.py --dataset='pokec_z' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=0 --d='yes'
python in-train.py --dataset='pokec_z' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='no'

###pre vs. raw
# gcn
python reweight.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0.5 --alpha=2
python pretrain.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=27 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0.5 --m_epoch=100 --alpha=2

python reweight.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.25
python reweight.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=100 --dropout=0.2 --delta=1 --c_lr=0.15 --e_lr=0.15 --c_wd=0.001 --e_wd=0.001
python pretrain.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=100 --dropout=0.2 --delta=1 --c_lr=0.15 --e_lr=0.15 --c_wd=0.001 --e_wd=0.001
python pretrain.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=100 --dropout=0.2 --delta=10 --c_lr=0.15 --e_lr=0.15 --c_wd=0.001 --e_wd=0.001

python reweight.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --dropout=0.2 --delta=0.2 --c_lr=0.1 --e_lr=0.1 --c_wd=0.001 --e_wd=0.001
python pretrain.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --m_epoch=100 --dropout=0.2 --delta=0.2 --c_lr=0.1 --e_lr=0.1 --c_wd=0.001 --e_wd=0.001

#GIN
python reweight.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0.75
python pretrain.py --dataset='german' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=27 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0.75 --m_epoch=100 

python reweight.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --dropout=0.2 --delta=0.1 --c_lr=0.15 --e_lr=0.15  --c_wd=0.001 --e_wd=0.001
python pretrain.py --dataset='credit' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --m_epoch=50 --delta=0.1 --c_lr=0.15 --e_lr=0.15  --c_wd=0.001 --e_wd=0.001

python reweight.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --delta=0.1 --c_lr=0.1 --e_lr=0.1  --c_wd=0.001 --e_wd=0.001
python pretrain.py --dataset='bail' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --m_epoch=200 --delta=0.1 --c_lr=0.2 --e_lr=0.2  --c_wd=0.001 --e_wd=0.001

#SAGE
python reweight.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=54 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0.25 --alpha=3
python pretrain.py --dataset='german' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=27 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=0.25 --m_epoch=100 --alpha=3

python reweight.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --dropout=0.2 --delta=1.5 --c_lr=0.15 --e_lr=0.15  --c_wd=0.001 --e_wd=0.001
python pretrain.py --dataset='credit' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --dropout=0.2 --m_epoch=50 --delta=1.5 --c_lr=0.15 --e_lr=0.15  --c_wd=0.001 --e_wd=0.001

python reweight.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --dropout=0.8 --delta=0.05
python pretrain.py --dataset='bail' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=36 --epoch=150 --m_epoch=50 --dropout=0.8 --delta=0.05 

### P-score

python neutralization_proof.py --d_epochs=250  --dataset='german' --runs=1 --hidden=27 --d_lr=0.001 --delta=3 --seed=0
python neutralization_proof.py --d_epochs=20  --dataset='credit' --runs=1 --hidden=13  --d_lr=0.001 --delta=0.25 --seed=0
python neutralization_proof.py --d_epochs=30  --dataset='pokec_n' --runs=1 --hidden=266  --d_lr=0.001 --delta=0.5 --seed=0
python neutralization_proof.py --d_epochs=30  --dataset='pokec_z' --runs=1 --hidden=277  --d_lr=0.001 --delta=2 --seed=0

### Delta Tuning
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=0 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=0.5 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=1 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=2 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=5 --d='yes' --c_wd=0.001 --e_wd=0.001
python in-train.py --dataset='german' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=10 --d='yes' --c_wd=0.001 --e_wd=0.001

python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0 --d='yes'
python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=0.5 --d='yes'
python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=1 --d='yes'
python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=2 --d='yes'
python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=5 --d='yes'
python in-train.py --dataset='credit' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=13 --epoch=100  --delta=10 --d='yes'  

python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0.1 
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=0.5
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=1
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=2
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=5
python in-train.py --dataset='bail' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.01 --e_lr=0.01 --delta=10


### additional dataset pokec 
## If the performance is not optimal, you can try adjusting the epoch between 50 and 200.
# vanilla
python gnn.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.1 --e_lr=0.01
python gnn.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.1 --e_lr=0.01
python gnn.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.1 --e_lr=0.01

python gnn.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.1 --e_lr=0.01
python gnn.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.1 --e_lr=0.01
python gnn.py --dataset='pokec_z' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --alpha=0 --c_lr=0.1 --e_lr=0.01
### FairSIN + pokec
# pokec-n
python in-train.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=1 --d='yes'
python in-train.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='yes'

python in-train.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=1 --d='yes' 
python in-train.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=80 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='yes'  --m_lr=0.01
python in-train.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=80 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='yes' --alpha=0.5 --m_lr=0.01

python in-train.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=1 --d='yes' 
python in-train.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=50 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='yes'  --m_lr=0.01 

#pokec-z
python in-train.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='yes'
python in-train.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=0.5 --d='yes' --alpha=0.5 --m_lr=0.1

python in-train.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=1 --d='yes' 
python in-train.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=80 --c_lr=0.1 --e_lr=0.01 --delta=1 --d='yes' --alpha=0.5 --m_lr=0.01

python in-train.py --dataset='pokec_z' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.1 --delta=1 --d='yes' --alpha=3

### FairSIN-G + pokec
# pokec-n
python reweight.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python reweight.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python reweight.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
# pokec-z
python reweight.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python reweight.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1 --alpha=0.
python reweight.py --dataset='pokec_z' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 

### FairSIN-F + pokec
# pokec-n
python pretrain.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python pretrain.py --dataset='pokec_n' --encoder='GCN' --c_epochs=10 --runs=1 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.01 --delta=1.5 

python pretrain.py --dataset='pokec_n' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python pretrain.py --dataset='pokec_n' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
# pokec-z
python pretrain.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python pretrain.py --dataset='pokec_z' --encoder='GCN' --c_epochs=10 --runs=5 --hidden=18 --epoch=150 --c_lr=0.1 --e_lr=0.01 --delta=1.5 

python pretrain.py --dataset='pokec_z' --encoder='GIN' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
python pretrain.py --dataset='pokec_z' --encoder='SAGE' --c_epochs=10 --runs=5 --hidden=18 --epoch=100 --c_lr=0.1 --e_lr=0.1 --delta=1 
