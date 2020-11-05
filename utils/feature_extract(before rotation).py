import pandas as pd
import numpy as np
import math
from data_preprocess import get_unit_data,get_package_data
from feature_extraction_functions import step_divide,mean_sd_instep,mean_sd_step,step_based_skewness

#t prefix stands for training set and s stand for testing set.
# for the t group, they are (151,?) numpy arrays.
# for the s group, they are (38,?) arrays.
taccel,taccel_time,trotate,trotate_time,tpedo_time,saccel,saccel_time,srotate,srotate_time,spedo_time=get_package_data()

# generate a ,28 feature
training_rows=taccel.shape[0]
testing_rows=saccel.shape[0]
tfeatures=np.zeros(shape=(training_rows,28))
sfeatures=np.zeros(shape=(testing_rows,28))

#generate the steps_divided accel and time
tstep_divided_accel,tstep_divided_time=step_divide('t',tpedo_time,taccel,taccel_time)	   # step divided acceleration of training
sstep_divided_accel,sstep_divided_time=step_divide('s',spedo_time,saccel,saccel_time)	   # step divided acceleration of testing

tjerk=[]
tjerk_time=[]
for i in range(training_rows):
	
	tfeatures[i,0]=taccel[i].mean()                                                        # 0.accel mean
	tfeatures[i,1]=np.std(taccel[i])                                                       # 1.accel sd
	tfeatures[i,4]=np.sqrt(np.mean(np.square(taccel[i])))                                  # 4. accel root mean square of acceleration
	
	tjerk_temp=np.absolute(np.diff(taccel[i]))/np.diff(taccel_time[i])                     # jerk for a walk
	tjerk.append(tjerk_temp)	                                 		                   # the jerk of training_data
	tfeatures[i,5]=np.mean(tjerk_temp)                                                     # 5. the mean of jerk
	tfeatures[i,6]=np.std(tjerk_temp)													   # 6. the std of jerk	
	tjerk_time.append(taccel_time[i][1:len(taccel_time[i])])							   # the jerk time uses the 2- the last time of accel time
	tfeatures[i,9]=np.sqrt(np.mean(np.square(tjerk_temp)))								   # 9. jerk root mean square

tjerk=np.array(tjerk)
tjerk_time=np.array(tjerk_time)

sjerk=[]
sjerk_time=[]
for i in range(testing_rows):
	sfeatures[i,0]=saccel[i].mean()                                                        # 0.accel mean
	sfeatures[i,1]=np.std(saccel[i])                                                       # 1.accel sd
	sfeatures[i,4]=np.sqrt(np.mean(np.square(saccel[i])))                                  # 4. accel root mean square of acceleration
	
	sjerk_temp=np.absolute(np.diff(saccel[i]))/np.diff(saccel_time[i])                     # jerk for a walk
	sjerk.append(sjerk_temp)	                                 		                   # the jerk of testing_data
	sfeatures[i,5]=np.mean(sjerk_temp)                                                     # 5. the mean of jerk
	sfeatures[i,6]=np.std(sjerk_temp)													   # 6. the std of jerk
	sjerk_time.append(saccel_time[i][1:len(saccel_time[i])])							   # the jerk time of testing data
	sfeatures[i,9]=np.sqrt(np.mean(np.square(sjerk_temp)))								   # 9. jerk root mean square

sjerk=np.array(sjerk)
sjerk_time=np.array(sjerk_time)


# generate the steps_divided jerk and time
tstep_divided_jerk,tstep_divided_jerk_time=step_divide('t',tpedo_time,tjerk,tjerk_time)	   # step divided jerk of training
sstep_divided_jerk,sstep_divided_jerk_time=step_divide('s',spedo_time,sjerk,sjerk_time)	   # step divided jerk of testing


tfeatures[:,2],tfeatures[:,3]=mean_sd_instep(tstep_divided_accel)						   # 2. accel mean of max in each step
sfeatures[:,2],sfeatures[:,3]=mean_sd_instep(sstep_divided_accel)						   # 3. accel sd of max in each step

tfeatures[:,7],tfeatures[:,8]=mean_sd_instep(tstep_divided_jerk)						   # 7. jerk mean of max in each step
sfeatures[:,7],sfeatures[:,8]=mean_sd_instep(sstep_divided_jerk)						   # 8. jerk sd of max in each step

tfeatures[:,10],tfeatures[:,11]=mean_sd_step(tstep_divided_time)						   # 10, the mean of duration of steps of a walk
sfeatures[:,10],sfeatures[:,11]=mean_sd_step(sstep_divided_time)						   # 11. the sd of duration of steps of a walk

tfeatures[:,12]=step_based_skewness(tstep_divided_accel)								   # 12. the skewness based on each step
sfeatures[:,12]=step_based_skewness(sstep_divided_accel)

tfeatures[:,13]=step_based_kurtosis(tstep_divided_accel)								   # 13. the kurtosis based on each step
sfeatures[:,13]=step_based_kurtosis(sstep_divided_accel)
