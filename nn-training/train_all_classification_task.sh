echo "Training model on each dataset"

# Shallow TCNs

for j in {0..50..1}  #20
    do 
    echo "Attempt $i "
    echo "Index $j "
    let "end_idx=$j+1"

    # exp id 4015
    python3.6 train_all_classification_task.py --type conv --arch_type spatial_temporal --exp_id 245 --start_id $j --end_id $end_idx
    python3.6 train_all_classification_task.py --type conv --arch_type temporal_spatial --exp_id 245 --start_id $j --end_id $end_idx
    python3.6 train_all_classification_task.py --type conv --arch_type spatiotemporal --exp_id 245 --start_id $j --end_id $end_idx

done


# Deeper TCNs

for j in {0..50..1}  #20
    do 
    echo "Attempt $i "
    echo "Index $j "
    let "end_idx=$j+1"

    # exp id 5015
    python3.6 train_all_classification_task.py --type conv_new --arch_type spatial_temporal --exp_id 255 --start_id $j --end_id $end_idx
    python3.6 train_all_classification_task.py --type conv_new --arch_type temporal_spatial --exp_id 255 --start_id $j --end_id $end_idx
    python3.6 train_all_classification_task.py --type conv_new --arch_type spatiotemporal --exp_id 255 --start_id $j --end_id $end_idx

done