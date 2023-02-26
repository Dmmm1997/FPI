name="olddata_timm_Pvtv2_CCN_MGFHeadAVG_400_128_nw15_R5"
train_dir="/home/dmmm/Dataset/FPI/FPI2022/train"
val_dir="/home/dmmm/Dataset/FPI/FPI2023/val"
num_worker=8
UAVhw=128
Satellitehw=400
batchsize=8
centerR=7
lr=0.0003
backbone="Pvt_small"
neg_weight=15
share=0
debug=1
head_pool="avg"
head="MultiGroupFusionHead"
neck="FPN"
python train.py --name $name --centerR $centerR --UAVhw $UAVhw --Satellitehw $Satellitehw \
                --num_worker $num_worker --backbone $backbone --batchsize $batchsize \
                --lr $lr --neg_weight $neg_weight --share $share --train_dir $train_dir --val_dir $val_dir \
                --debug $debug --head_pool $head_pool --neck $neck --head $head
        
test_dir="/home/dmmm/Dataset/FPI/FPI2022/test"
# mode="2019_2022_satellitemap_700-1800_cr0.9_stride100"
mode="merge_test_700-1800_cr0.95_stride100"
checkpoint="net_016.pth"
cd checkpoints/$name
python test_meter.py --checkpoint $checkpoint --test_dir $test_dir --mode $mode
cd ../../
