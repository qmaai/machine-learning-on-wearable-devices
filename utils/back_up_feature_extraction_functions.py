import numpy

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

def mean_sd_step(step_divide_time):														   # the input is the t/sstep_divided_time obtained from step_divide
	mean=[]
	sd=[]
	for i in range(len(step_divide_time)):
		walk_step_time=[]
		last_time=step_divide_time[i][0][0]
		for j in range(len(step_divide_time[i])):			
			walk_step_time.append(step_divide_time[i][j][-1]-last_time)					   # the duration of a step is calculated as the last index of step
			last_time=step_divide_time[i][j][-1]										   # minus the last index of last step
		walk_step_time=np.array(walk_step_time)

		mean.append(np.mean(walk_step_time))
		sd.append(np.std(walk_step_time))
	return mean,sd

def step_based_skewness(step_divided):
	skewness=[]
	for i in range(len(step_divided)):
		walk_step_skewness=[]
		for j in range(len(step_divided[i])):
			step=np.array(step_divided[i][j])
			step_minus_mean=step-np.mean(step)
			nominator=np.mean(np.power(step_minus_mean,3))
			denominator=np.power(np.mean(np.power(step_minus_mean,2)),1.5)
			walk_step_skewness.append(nominator/denominator)
		skewness.append(np.mean(np.array(walk_step_skewness)))
	return skewness