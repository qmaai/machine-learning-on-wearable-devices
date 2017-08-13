import numpy as np

def step_based_kurtosis(step_divided):
	kurtosis=[]
	for i in range(len(step_divided)):
		walk_step_kurtosis=[]
		for j in range(len(step_divided[i])):
			if len(step_divided[i][j])<2:
				continue
			step=np.array(step_divided[i][j])
			step_minus_mean=step-np.mean(step)
			nominator=np.mean(np.power(step_minus_mean,4))
			denominator=np.power(np.mean(np.power(step_minus_mean,2)),2)
			walk_step_kurtosis.append(nominator/denominator)
		kurtosis.append(np.mean(np.array(walk_step_kurtosis)))
	return kurtosis
'''
features (each line represent both mean and sd):
acceleration:
	0. accel mean
	1. accel sd
	2. accel mean of max in each step
	3. accel sd of max in each step
	4. accel modified mean: root mean square

	5. jerk mean
	6. jerk sd5
	7. jerk mean of max in each step
	8. jerk sd of max in each step
	9.jerk modified mean: root of mean squared accel

	10.step duration mean
	11.step duration sd
	12.skewness mean (the symmetry of the graph of each record of acceleration) 
rotation:3
	same with acceleration
'''
'''
walk_step_kurtosis=[]
for j in range(len(tstep_divided_accel[144])):
	if len(tstep_divided_accel[144][j])<2:
		continue
	step=np.array(tstep_divided_accel[144][j])
	step_minus_mean=step-np.mean(step)
	nominator=np.mean(np.power(step_minus_mean,4))
	denominator=np.power(np.mean(np.power(step_minus_mean,2)),2)
	print('nominator is '+str(nominator)+' and denominator is '+str(denominator))
	if(denominator==0):
		print(tstep_divided_accel[144][j])
	walk_step_kurtosis.append(nominator/denominator)
print(np.mean(np.array(walk_step_kurtosis)))

walk_step_skewness=[]
for j in range(len(tstep_divided_accel[143])):
	if len(tstep_divided_accel[143][j])<2:
		continue
	step=np.array(tstep_divided_accel[143][j])
	step_minus_mean=step-np.mean(step)
	nominator=np.mean(np.power(step_minus_mean,4))
	denominator=np.power(np.mean(np.power(step_minus_mean,2)),2)
	if(nominator==0 or denominator==0):
		print('j is '+str(j))
		print(tstep_divided_accel[143][j])
	walk_step_skewness.append(nominator/denominator)

walk_rotate_temp=[]
walk_time_temp=[]

pointer=0
for j in range(pedo_time[9].shape[0]):
	rotate_temp=[]
	time_temp=[]
	for z in range(pointer,rotate[9].shape[0]):
		if(rotate_time[i][pointer]<=pedo_time[i][j]):
			rotate_temp.append(rotate[i][pointer])
			time_temp.append(rotate_time[i][pointer])
			pointer=pointer+1
		else:
			if len(rotate_temp)>0:
				walk_rotate_temp.append(rotate_temp)
				walk_time_temp.append(time_temp)
				pointer=pointer+1
			break


kurtosis=[]
walk_step_kurtosis=[]
for j in range(len(tstep_divided_rotate[9])):
	if len(tstep_divided_rotate[i][j])<2:
		continue
	step=np.array(tstep_divided_rotate[9][j])
	step_minus_mean=step-np.mean(step)
	nominator=np.mean(np.power(step_minus_mean,4))
	denominator=np.power(np.mean(np.power(step_minus_mean,2)),2)
	walk_step_kurtosis.append(nominator/denominator)
print(np.mean(np.array(walk_step_kurtosis)))

for i in range(len(tfeatures)):
	for j in range()

tstep_accel=[]
tstep_time=[]
for i in range(training_rows):
	walk_accel_temp=[]
	walk_time_temp=[]
	pointer=0
	for j in range(tpedo_time[i].shape[0]):
		accel_temp=[]
		time_temp=[]
		for z in range(pointer,taccel[i].shape[0]):
			if(taccel_time[i][pointer]<=tpedo_time[i][j]):
				accel_temp.append(taccel[i][pointer])
				time_temp.append(taccel_time[i][pointer])
				pointer=pointer+1
			else:
				walk_accel_temp.append(accel_temp)
				walk_time_temp.append(time_temp)
				pointer=pointer+1
				break
	tstep_accel.append(walk_accel_temp)
	tstep_time.append(walk_time_temp)


def step_divide_accel(pedo_time,accel,accel_time):
	step_accel=[]
	step_time=[]
	rows=accel.shape[0]
	for i in range(rows):
		walk_accel_temp=[]
		walk_time_temp=[]
		pointer=0
		for j in range(pedo_time[i].shape[0]):
			accel_temp=[]
			time_temp=[]
			for z in range(pointer,accel[i].shape[0]):
				if(accel_time[i][pointer]<=pedo_time[i][j]):
					accel_temp.append(accel[i][pointer])
					time_temp.append(accel_time[i][pointer])
					pointer=pointer+1
				else:
					if len(accel_temp)>0:
						walk_accel_temp.append(accel_temp)
						walk_time_temp.append(time_temp)
						pointer=pointer+1
					break
		step_accel.append(walk_accel_temp)
		step_time.append(walk_time_temp)
	return step_accel,step_time


skewness=[]
for i in range(len(tstep_divided_accel)):
	walk_step_skewness=[]
	for j in range(len(tstep_divided_accel[i])):
		if len(tstep_divided_accel[i][j])<2:
			continue
		step=np.array(tstep_divided_accel[i][j])
		step_minus_mean=step-np.mean(step)
		nominator=np.mean(np.power(step_minus_mean,3))
		denominator=np.power(np.mean(np.power(step_minus_mean,2)),1.5)
		walk_step_skewness.append(nominator/denominator)
	skewness.append(np.mean(np.array(walk_step_skewness)))

for i in range(len(taccel)):
	if len(taccel[i])>200:
		print(i)
tAccelerometer=[30,32,33,34,35,82,86,87,91,9,93,94,95]
for i in range(len(saccel)):
	if len(saccel[i])>200:
		print(i)
sAccelerometer=[8,20,22,23]
'''
def max_freq_ampl(accel_or_rotate,t_or_s,array):
	max_amp=[]
	max_freq=[]
	if(accel_or_rotate=='a'):
		if(t_or_s=='t'):
			Accelerometer=[0,32,33,34,35,82,86,87,91,9,93,94,95]
		else:
			Accelerometer=[8,20,22,23]
		for i in range(len(array)):
			if i in Accelerometer:
				timestep=0.066666667
			else:
				timestep=0.2
			fourier_temp=np.fft.fft(array[i])
			freq_temp=np.fft.fftfreq(len(array[i]),timestep)
			max_amp.append(np.max(fourier_temp))
			max_freq.append(np.max(freq_temp))
	else:
		timestep=0.2
		for i in range(len(array)):
			fourier_temp=np.fft.fft(array[i])
			freq_temp=np.fft.fftfreq(len(array[i]),timestep)
			max_amp.append(np.max(fourier_temp))
			max_freq.append(np.max(freq_temp))
	return max_amp,ma_freq


