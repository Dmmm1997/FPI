name="repeat_lr0.00015_B8_VitS_400_112_nw15_R2"
num_worker=8
UAVhw=112
Satellitehw=400
batchsize=8
centerR=2
lr=0.00015
backbone="Deit-S"
neg_weight=15
share=0
python train.py --name $name --centerR $centerR --UAVhw $UAVhw --Satellitehw $Satellitehw \
                --num_worker $num_worker --backbone $backbone --batchsize $batchsize \
                --lr $lr --neg_weight $neg_weight --share $share

cd checkpoints/$name
checkpoint="net_016.pth"
python test_meter.py --checkpoint $checkpoint
cd ../../


name="repeat_lr0.00015_B8_VitS_400_112_nw15_R3"
num_worker=8
UAVhw=112
Satellitehw=400
batchsize=8
centerR=3
lr=0.00015
backbone="Deit-S"
neg_weight=15
share=0
python train.py --name $name --centerR $centerR --UAVhw $UAVhw --Satellitehw $Satellitehw \
                --num_worker $num_worker --backbone $backbone --batchsize $batchsize \
                --lr $lr --neg_weight $neg_weight --share $share

cd checkpoints/$name
checkpoint="net_016.pth"
python test_meter.py --checkpoint $checkpoint
cd ../../


name="repeat_lr0.00015_B8_VitS_400_112_nw15_R5"
num_worker=8
UAVhw=112
Satellitehw=400
batchsize=8
centerR=5
lr=0.00015
backbone="Deit-S"
neg_weight=15
share=0
python train.py --name $name --centerR $centerR --UAVhw $UAVhw --Satellitehw $Satellitehw \
                --num_worker $num_worker --backbone $backbone --batchsize $batchsize \
                --lr $lr --neg_weight $neg_weight --share $share

cd checkpoints/$name
checkpoint="net_016.pth"
python test_meter.py --checkpoint $checkpoint
cd ../../



name="repeat_lr0.00015_B8_VitS_400_112_nw15_R7"
num_worker=8
UAVhw=112
Satellitehw=400
batchsize=8
centerR=7
lr=0.00015
backbone="Deit-S"
neg_weight=15
share=0
python train.py --name $name --centerR $centerR --UAVhw $UAVhw --Satellitehw $Satellitehw \
                --num_worker $num_worker --backbone $backbone --batchsize $batchsize \
                --lr $lr --neg_weight $neg_weight --share $share

cd checkpoints/$name
checkpoint="net_016.pth"
python test_meter.py --checkpoint $checkpoint
cd ../../