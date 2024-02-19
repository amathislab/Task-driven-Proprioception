echo "Training model on each dataset"

# Shallow TCNs

for j in {0..50..1}  #20
    do 
    echo "Attempt $i "
    echo "Index $j "
    let "end_idx=$j+1"

    ## exp_id 10015
    python3.6 train_all_barlow_task.py --type barlow_conv --arch_type spatial_temporal --exp_id 305 --start_id $j --end_id $end_idx
    python3.6 train_all_barlow_task.py --type barlow_conv --arch_type temporal_spatial --exp_id 305 --start_id $j --end_id $end_idx
    python3.6 train_all_barlow_task.py --type barlow_conv --arch_type spatiotemporal --exp_id 305 --start_id $j --end_id $end_idx

done


# Deeper TCNs

for j in {0..50..1}  #20
    do 
    echo "Attempt $i "
    echo "Index $j "
    let "end_idx=$j+1"

    ## exp_id 10030
    python3.6 train_all_barlow_task.py --type barlow_conv_new --arch_type spatial_temporal --exp_id 330 --start_id $j --end_id $end_idx
    python3.6 train_all_barlow_task.py --type barlow_conv_new --arch_type temporal_spatial --exp_id 330 --start_id $j --end_id $end_idx
    python3.6 train_all_barlow_task.py --type barlow_conv_new --arch_type spatiotemporal --exp_id 330 --start_id $j --end_id $end_idx

done