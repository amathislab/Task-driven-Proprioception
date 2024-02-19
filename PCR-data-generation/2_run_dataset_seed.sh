#!/bin/bash

########### Bash script to generate synthetic muscle spindle inputs for each trajectory ############

echo "Generating trajectories"

## j is the folder number where the trajectory will be saved
## i is the character number (From 1 to 20 single-stroke character).

## Each character will generate two folders with the number that is even for horizontal and odd for vertical.
## E.g. character a (i=1) --> 0 horizontal folder and 1 vertical folder

## All monkey names [snap, butter, lando, han01_05, han11_22, chips]

for j in {9000..9000..1}
	do
	
	SEED=$RANDOM
	for i in {1..20..1}  #20

	  do 
		 echo "sample $j char $i "
		 let "val=$i*2"
		 let "val2=$val-1"
		 python3.6 generate_data_hor_seed.py --label $i --plane 'horizontal' --monkey_name 'snap' --seed $SEED $val2 $j 
		 python3.6 generate_data_ver_seed.py --label $i --plane 'vertical' --monkey_name 'snap' --seed $SEED $val $j

	 done
done