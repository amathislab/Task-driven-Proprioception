echo "Training model on each dataset"

########## REGRESS

### Shallow TCNs

### If you need to change the target keys:
# Hand related targets:         'endeffector_coords',   'endeffector_vel',      'endeffector_acc'
# Elbow related targets:        'elbow_coords',         'elbow_vel',            'elbow_acc'
# Joint related targets:        'joint_coords',         'joint_vel',            'joint_acc'


### The following in an example for training TCNs to predict joint position and velocity.

for j in {0..50..1}  #20
        do 
        echo "Attempt $i "
        echo "Index $j "
        let "end_idx=$j+1"

        ## exp_id 17416
        python3.6 train_all_regression_tasks.py --type conv --arch_type spatial_temporal \
                                                --exp_id 616 --start_id $j --end_id $end_idx \
                                                --task regress_joints_pos_vel \
                                                --target_keys joint_coords joint_vel


        python3.6 train_all_regression_tasks.py --type conv --arch_type temporal_spatial \
                                                --exp_id 616 --start_id $j --end_id $end_idx \
                                                --task regress_joints_pos_vel \
                                                --target_keys joint_coords joint_vel


        python3.6 train_all_regression_tasks.py --type conv --arch_type spatiotemporal \
                                                --exp_id 616 --start_id $j --end_id $end_idx \
                                                --task regress_joints_pos_vel \
                                                --target_keys joint_coords joint_vel

done

### Deeper TCNs

for j in {0..50..1}  #20
        do 
        echo "Attempt $i "
        echo "Index $j "
        let "end_idx=$j+1"
        
        # exp_id 17431
        python3.6 train_all_regression_tasks.py --type conv_new --arch_type spatial_temporal \
                                                --exp_id 631 --start_id $j --end_id $end_idx \
                                                --task regress_joints_pos_vel \
                                                --target_keys joint_coords joint_vel


        python3.6 train_all_regression_tasks.py --type conv_new --arch_type temporal_spatial \
                                                --exp_id 631 --start_id $j --end_id $end_idx \
                                                --task regress_joints_pos_vel \
                                                --target_keys joint_coords joint_vel


        python3.6 train_all_regression_tasks.py --type conv_new --arch_type spatiotemporal \
                                                --exp_id 631 --start_id $j --end_id $end_idx \
                                                --task regress_joints_pos_vel \
                                                --target_keys joint_coords joint_vel

done