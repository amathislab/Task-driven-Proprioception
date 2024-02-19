echo "Training model on each dataset"

########## AUTOENCODER TASK

for j in {0..49..1}  #20
	do 
	echo "Attempt $i "
	echo "Index $j "
	let "end_idx=$j+1"

	#Exp_id 20716
	python3.6 train_all_autoencoder_task.py --type conv --arch_type spatial_temporal --exp_id 516 --start_id $j --end_id $end_idx --task autoencoder
	python3.6 train_all_autoencoder_task.py --type conv --arch_type temporal_spatial --exp_id 516 --start_id $j --end_id $end_idx --task autoencoder
	python3.6 train_all_autoencoder_task.py --type conv --arch_type spatiotemporal --exp_id 516 --start_id $j --end_id $end_idx --task autoencoder
	done

for j in {0..49..1}  #20
	do 
	echo "Attempt $i "
	echo "Index $j "
	let "end_idx=$j+1"

	#exp_id 20731
	python3.6 train_all_autoencoder_task.py --type conv_new --arch_type spatial_temporal --exp_id 531 --start_id $j --end_id $end_idx --task autoencoder
	python3.6 train_all_autoencoder_task.py --type conv_new --arch_type temporal_spatial --exp_id 531 --start_id $j --end_id $end_idx --task autoencoder
	python3.6 train_all_autoencoder_task.py --type conv_new --arch_type spatiotemporal --exp_id 531 --start_id $j --end_id $end_idx --task autoencoder
	done