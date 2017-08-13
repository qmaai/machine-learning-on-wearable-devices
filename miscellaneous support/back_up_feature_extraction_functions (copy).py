import numpy as np
import heapq
from scipy import signal
# input: numpy array
# use the pedo_time to compare with accel_time
# split the walk of data into many steps inside the walk
# output will be a (x,y,z) dimensional list.
# x is the number of walks; y the number of steps in each walk; z the data in each steps
def step_divide(training_or_testing,pedo_time,accel,accel_time):
	step_accel=[]
	step_time=[]
	rows=accel.shape[0]
	if training_or_testing=='t':
		malvalue=[9,16,75]
	else:
		malvalue=[2,3]
	for i in range(rows):
		walk_accel_temp=[]
		walk_time_temp=[]
		## these three only got 1 step counter data.
		## I am quite displeased about sensorFetch
		## should have locked the sensors selection tightly!
		if(i in malvalue):
			for j in range(0,accel[i].shape[0]-5,5):
				walk_accel_temp.append([accel[i][j],accel[i][j+1],accel[i][j+2],accel[i][j+3],accel[i][j+4]])
				walk_time_temp.append([accel_time[i][j],accel_time[i][j+1],accel_time[i][j+2],accel_time[i][j+3],accel_time[i][j+4]])
		else:
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

# calculate the mean and the sd of the max of steps
# for feature 2/3(accel)/7/8(jerk)
def mean_sd_instep(step_divided_array):													   # the input of the file is a 3 dimensional list
	mean_of_max=[]
	sd_of_max=[]
	for i in range(len(step_divided_array)):											   # go inside each walk
		steps_max=[]																	   # for wach walk construct a max of steps list
		for j in range(len(step_divided_array[i])):										   # go inside each steps
			steps_max.append(max(step_divided_array[i][j]))								   # get the max inside the step and append it to the walk list
		steps_max=np.array(steps_max)													   # turn it into a numpy array

		mean_of_max.append(np.mean(steps_max))
		sd_of_max.append(np.std(steps_max))
	return mean_of_max,sd_of_max

# calculate the mean and sd of the duration of steps
# for function 10/11 steps duration mean/sd
def mean_sd_step(step_divide_time):														   # the input is the t/sstep_divided_time obtained from step_divide
	mean=[]
	sd=[]
	max_list=[]
	min_list=[]
	median_list=[]
	for i in range(len(step_divide_time)):
		walk_step_time=[]
		last_time=step_divide_time[i][0][0]
		for j in range(len(step_divide_time[i])):			
			walk_step_time.append(step_divide_time[i][j][-1]-last_time)					   # the duration of a step is calculated as the last index of step
			last_time=step_divide_time[i][j][-1]										   # minus the last index of last step
		walk_step_time=np.array(walk_step_time)

		mean.append(np.mean(walk_step_time))
		sd.append(np.std(walk_step_time))
		max_list.append(np.max(walk_step_time))
		min_list.append(np.min(walk_step_time))
		median_list.append(np.median(walk_step_time))
	return mean,sd,max_list,min_list,median_list

# calculate skewness. for feature 12
def step_based_skewness(step_divided):
	skewness=[]
	for i in range(len(step_divided)):
		walk_step_skewness=[]
		for j in range(len(step_divided[i])):
			if len(step_divided[i][j])<3:
				continue
			step=np.array(step_divided[i][j])
			step_minus_mean=step-np.mean(step)
			nominator=np.mean(np.power(step_minus_mean,3))
			denominator=np.power(np.mean(np.power(step_minus_mean,2)),1.5)
			walk_step_skewness.append(nominator/denominator)
		skewness.append(np.mean(np.array(walk_step_skewness)))
	return np.nan_to_num(skewness)

# calculate kurtosis for feature 13
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

# for calculating the max, min and mean of relevant input
def max_min_median(array):
	max_list=[]
	min_list=[]
	median_list=[]
	for i in range(len(array)):
		max_list.append(np.max(array[i]))
		min_list.append(np.max(array[i]))
		median_list.append(np.max(array[i]))
	return max_list,min_list,median_list

# input: accel_or_rotate 'a' for accel or jerk; 'r' for rotate and angu
# input: t_or_s 't' for training and 's' for testing
def max_freq_ampl(accel_or_rotate,t_or_s,array):
	max_amp=[]
	max_freq=[]
	most_important_freq=[]
	sec_important_freq=[]
	freq_span=[]
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
			# the amplitude of frequencies
			fourier_temp=np.fft.fft(array[i])
			# the frequencies
			freq_temp=np.fft.fftfreq(len(array[i]),timestep)
			# the max amplitude of all frequencies
			max_amp.append(np.max(fourier_temp))
			# the max existing frequencies
			max_freq.append(np.max(freq_temp))
			# where is the largest amplitude
			most_important_index=np.argmax(fourier_temp)
			# the most important frequency
			freq1=np.copy(freq_temp[most_important_index])
			most_important_freq.append(freq1)
			# delete the most important index and the frequency
			fourier_temp=np.delete(fourier_temp,most_important_index,0)
			freq_temp=np.delete(freq_temp,most_important_index,0)
			# find the second largest
			most_important_index=np.argmax(fourier_temp)
			freq2=freq_temp[most_important_index]
			sec_important_freq.append(freq2)
			# the gap between
			freq_span.append(np.absolute(freq1-freq2))

	else:
		timestep=0.2
		for i in range(len(array)):
			# the amplitude of frequencies
			fourier_temp=np.fft.fft(array[i])
			# the frequencies
			freq_temp=np.fft.fftfreq(len(array[i]),timestep)
			# the max amplitude of all frequencies
			max_amp.append(np.max(fourier_temp))
			# the max existing frequencies
			max_freq.append(np.max(freq_temp))
			# where is the largest amplitude
			most_important_index=np.argmax(fourier_temp)
			# the most important frequency
			freq1=np.copy(freq_temp[most_important_index])
			most_important_freq.append(freq1)
			# delete the most important index and the frequency
			fourier_temp=np.delete(fourier_temp,most_important_index,0)
			freq_temp=np.delete(freq_temp,most_important_index,0)
			# find the second largest
			most_important_index=np.argmax(fourier_temp)
			freq2=freq_temp[most_important_index]
			sec_important_freq.append(freq2)
			# the gap between
			freq_span.append(np.absolute(freq1-freq2))
	return max_amp,max_freq,most_important_freq,sec_important_freq,freq_span

def psd(accel_or_rotate,t_or_s,array):
	psd_mean=[]
	psd_std=[]
	if(accel_or_rotate=='a'):
		if(t_or_s=='t'):
			Accelerometer=[0,32,33,34,35,82,86,87,91,9,93,94,95]
		else:
			Accelerometer=[8,20,22,23]
		for i in range(len(array)):
			if i in Accelerometer:
				sampling_freq=15
			else:
				sampling_freq=5
			f,psd=signal.periodogram(array[i],sampling_freq)
			psd_mean.append(np.mean(psd))
			psd_std.append(np.std(psd))
	else:
		sampling_freq=5
		for i in range(len(array)):
			f,psd=signal.periodogram(array[i],sampling_freq)
			psd_mean.append(np.mean(psd))
			psd_std.append(np.std(psd))
	return psd_mean,psd_std