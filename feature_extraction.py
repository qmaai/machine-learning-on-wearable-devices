import pandas as pd
import numpy as np
import math
import os
from data_preprocess import get_unit_data,get_package_data
from feature_extraction_functions import step_divide,mean_sd_instep,mean_sd_step,step_based_skewness, \
											step_based_kurtosis,max_min_median,max_freq_ampl,psd

save_directory='feature_saved_directory'
tfeature_file_name=save_directory+'/tfeature_raw'
sfeature_file_name=save_directory+'/sfeature_raw'

#t prefix stands for training set and s stand for testing set.
# for the t group, they are (151,?) numpy arrays.
# for the s group, they are (38,?) arrays.
taccel,taccel_time,trotate,trotate_time,tpedo_time,saccel,saccel_time,srotate,srotate_time,spedo_time=get_package_data()

# generate a ,28 feature
training_rows=taccel.shape[0]
testing_rows=saccel.shape[0]
tfeatures=np.zeros(shape=(training_rows,69))
sfeatures=np.zeros(shape=(testing_rows,69))

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

tfeatures[:,10],tfeatures[:,11],tfeatures[:,40],tfeatures[:,41],tfeatures[:,42]=mean_sd_step(tstep_divided_time) # the mean, sd, max, min, median of step
sfeatures[:,10],sfeatures[:,11],sfeatures[:,40],sfeatures[:,41],sfeatures[:,42]=mean_sd_step(sstep_divided_time) # duration of a walk

tfeatures[:,12]=step_based_skewness(tstep_divided_accel)								   # 12. the skewness based on each step
sfeatures[:,12]=step_based_skewness(sstep_divided_accel)

tfeatures[:,13]=step_based_kurtosis(tstep_divided_accel)								   # 13. the kurtosis based on each step
sfeatures[:,13]=step_based_kurtosis(sstep_divided_accel)


#generate the steps_divided rotate and time
tstep_divided_rotate,tstep_divided_rotate_time=step_divide('t',tpedo_time,trotate,trotate_time)	   # step divided rotatation of training
sstep_divided_rotate,sstep_divided_rotate_time=step_divide('s',spedo_time,srotate,srotate_time)	   # step divided rotatation of testing

tangu=[]
tangu_time=[]
for i in range(training_rows):
	
	tfeatures[i,14]=trotate[i].mean()                                                        # 0.rotate mean
	tfeatures[i,15]=np.std(trotate[i])                                                       # 1.rotate sd
	tfeatures[i,18]=np.sqrt(np.mean(np.square(trotate[i])))                                  # 4. rotate root mean square of rotateeration
	
	tangu_temp=np.absolute(np.diff(trotate[i]))/np.diff(trotate_time[i])                     # angu for a walk
	tangu.append(tangu_temp)	                                 		                   # the angu of training_data
	tfeatures[i,19]=np.mean(tangu_temp)                                                     # 5. the mean of angu
	tfeatures[i,20]=np.std(tangu_temp)													   # 6. the std of angu	
	tangu_time.append(trotate_time[i][1:len(trotate_time[i])])							   # the angu time uses the 2- the last time of rotate time
	tfeatures[i,23]=np.sqrt(np.mean(np.square(tangu_temp)))								   # 9. angu root mean square

tangu=np.array(tangu)
tangu_time=np.array(tangu_time)

sangu=[]
sangu_time=[]
for i in range(testing_rows):
	sfeatures[i,14]=srotate[i].mean()                                                        # 0.rotate mean
	sfeatures[i,15]=np.std(srotate[i])                                                       # 1.rotate sd
	sfeatures[i,18]=np.sqrt(np.mean(np.square(srotate[i])))                                  # 4. rotate root mean square of rotateeration
	
	sangu_temp=np.absolute(np.diff(srotate[i]))/np.diff(srotate_time[i])                     # angu for a walk
	sangu.append(sangu_temp)	                                 		                   # the angu of testing_data
	sfeatures[i,19]=np.mean(sangu_temp)                                                     # 5. the mean of angu
	sfeatures[i,20]=np.std(sangu_temp)													   # 6. the std of angu
	sangu_time.append(srotate_time[i][1:len(srotate_time[i])])							   # the angu time of testing data
	sfeatures[i,23]=np.sqrt(np.mean(np.square(sangu_temp)))								   # 9. angu root mean square

sangu=np.array(sangu)
sangu_time=np.array(sangu_time)


# generate the steps_divided angu and time
tstep_divided_angu,tstep_divided_angu_time=step_divide('t',tpedo_time,tangu,tangu_time)	   # step divided angu of training
sstep_divided_angu,sstep_divided_angu_time=step_divide('s',spedo_time,sangu,sangu_time)	   # step divided angu of testing


tfeatures[:,16],tfeatures[:,17]=mean_sd_instep(tstep_divided_rotate)					  
sfeatures[:,16],sfeatures[:,17]=mean_sd_instep(sstep_divided_rotate)					  

tfeatures[:,21],tfeatures[:,22]=mean_sd_instep(tstep_divided_angu)						   
sfeatures[:,21],sfeatures[:,22]=mean_sd_instep(sstep_divided_angu)						   

tfeatures[:,24]=step_based_skewness(tstep_divided_rotate)								  
sfeatures[:,24]=step_based_skewness(sstep_divided_rotate)

tfeatures[:,25]=step_based_kurtosis(tstep_divided_rotate)								   
sfeatures[:,25]=step_based_kurtosis(sstep_divided_rotate)

tfeatures[:,26],tfeatures[:,27],tfeatures[:,28]=max_min_median(taccel)
sfeatures[:,26],sfeatures[:,27],sfeatures[:,28]=max_min_median(saccel)

tfeatures[:,29],tfeatures[:,30],tfeatures[:,31]=max_min_median(tjerk)
sfeatures[:,29],sfeatures[:,30],sfeatures[:,31]=max_min_median(sjerk)

tfeatures[:,32],tfeatures[:,33],tfeatures[:,34]=max_min_median(trotate)
sfeatures[:,32],sfeatures[:,33],sfeatures[:,34]=max_min_median(srotate)

tfeatures[:,35],tfeatures[:,36],tfeatures[:,37]=max_min_median(tangu)
sfeatures[:,35],sfeatures[:,36],sfeatures[:,37]=max_min_median(sangu)

tfeatures[:,38]=step_based_skewness(tstep_divided_jerk)								   
sfeatures[:,38]=step_based_skewness(sstep_divided_jerk)

tfeatures[:,39]=step_based_kurtosis(tstep_divided_jerk)								
sfeatures[:,39]=step_based_kurtosis(sstep_divided_jerk)

tfeatures[:,43]=step_based_skewness(tstep_divided_angu)								   
sfeatures[:,43]=step_based_skewness(sstep_divided_angu)

tfeatures[:,44]=step_based_kurtosis(tstep_divided_angu)								
sfeatures[:,44]=step_based_kurtosis(sstep_divided_angu)

tfeatures[:,45],tfeatures[:,46],tfeatures[:,47],tfeatures[:,48],tfeatures[:,49]=max_freq_ampl('a','t',taccel)
sfeatures[:,45],sfeatures[:,46],sfeatures[:,47],sfeatures[:,48],sfeatures[:,49]=max_freq_ampl('a','s',saccel)

tfeatures[:,50],tfeatures[:,51],tfeatures[:,52],tfeatures[:,53],tfeatures[:,54]=max_freq_ampl('a','t',tjerk)
sfeatures[:,50],sfeatures[:,51],sfeatures[:,52],sfeatures[:,53],sfeatures[:,54]=max_freq_ampl('a','s',sjerk)

tfeatures[:,55],tfeatures[:,56],tfeatures[:,57],tfeatures[:,58],tfeatures[:,59]=max_freq_ampl('r','t',trotate)
sfeatures[:,55],sfeatures[:,56],sfeatures[:,57],sfeatures[:,58],sfeatures[:,59]=max_freq_ampl('r','s',srotate)

tfeatures[:,60],tfeatures[:,61],tfeatures[:,62],tfeatures[:,63],tfeatures[:,64]=max_freq_ampl('r','t',tangu)
sfeatures[:,60],sfeatures[:,61],sfeatures[:,62],sfeatures[:,63],sfeatures[:,64]=max_freq_ampl('r','s',sangu)

tfeatures[:,65],tfeatures[:,66]=psd('a','t',taccel)
sfeatures[:,65],sfeatures[:,66]=psd('a','s',saccel)

tfeatures[:,67],tfeatures[:,68]=psd('r','t',trotate)
sfeatures[:,67],sfeatures[:,68]=psd('r','s',srotate)

if not os.path.isdir(save_directory):
	os.mkdir(save_directory)

np.save(tfeature_file_name,tfeatures)
np.save(sfeature_file_name,sfeatures)

