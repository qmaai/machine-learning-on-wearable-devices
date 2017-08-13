import pandas as pd
import numpy as np
import math
import os
import random


# the function use os to step through all the files in data_work in order and implement the get_unit_data one by one.
def get_package_data():
	taccel=[]
	taccel_time=[]
	trotate=[]
	trotate_time=[]
	tpedo_time=[]
	saccel=[]
	saccel_time=[]
	srotate=[]
	srotate_time=[]
	spedo_time=[]
	for root,dirs,files in os.walk('data_work'):
		dirs.sort()
		for dir in dirs:
			for root_next,dirs_next,files_next in os.walk(os.path.join(root,dir)):
				files_next.sort()
				print(root_next)
				for file_next in files_next:					
					print(root_next+'/'+file_next)
					uaccel,urotate,uaccel_time,urotate_time,upedo_time=get_unit_data(root_next+'/'+file_next)
					if 'test' not in file_next :
						taccel.append(uaccel)
						taccel_time.append(uaccel_time)
						trotate.append(urotate)
						trotate_time.append(urotate_time)
						tpedo_time.append(upedo_time)
					else :
						saccel.append(uaccel)
						saccel_time.append(uaccel_time)
						srotate.append(urotate)
						srotate_time.append(urotate_time)
						spedo_time.append(upedo_time)
	return np.array(taccel),np.array(taccel_time),np.array(trotate),np.array(trotate_time),np.array(tpedo_time),np.array(saccel),np.array(saccel_time),np.array(srotate),np.array(srotate_time),np.array(spedo_time)


## a unit refers to one single csv file.
## the out put of the function is:
## acceleration (50,) rotation (50,) accel_time(50,) rotation_time(50,) pedo_time(?,)
## ? is the number of steps taken.
## input: absolute path of the csv file
## note that for each single csv file, start to get the data from one second after it starts
## end the data retrival from 11 seconds after it start.
def get_unit_data(file_path):
	df=pd.read_csv(file_path,usecols=[0,1,2,3,4],names=['time','sensor','x','y','z'])
	arr=df.as_matrix()
	wanted_rows=[]
	'''
	start_minute=arr[0,0].split(':')[1]
	start_second=arr[0,0].split(':')[2].split('.')[0]
	start_time=int(start_minute)*60+int(start_second)+1
	end_minute=arr[-1,0].split(':')[1]
	end_second=arr[-1,0].split(':')[2].split('.')[0]
	end_time=int(end_minute)*60+int(end_second)-1
	'''
	#initialise_sensor_wanted(arr)
	sensors_wanted=['Accelerometer','Gyroscope','Step Counter']
	for i in range(arr.shape[0]):
		if arr[i,1]=='Linear Acceleration':
			sensors_wanted=['Linear Acceleration','Gyroscope','Step Counter']
			break;
	for i in range(arr.shape[0]):
		if arr[i,1] in sensors_wanted:
			hour=arr[i,0].split(':')[0]
			minute=arr[i,0].split(':')[1]
			second=arr[i,0].split(':')[2].split('.')[0]
			time=int(hour)*60+int(minute)*60+int(second)
			#if (time>start_time and time<=end_time):
			millisecond=(int)(arr[i,0].split(':')[2].split('.')[1])
			time+=millisecond/1000
			arr[i,0]=time
			wanted_rows.append(arr[i,:])
	arr=np.array(wanted_rows)
	acceleration=[]
	rotation=[]
	time_accel=[]
	time_rotate=[]
	time_pedo=[]
	for i in range(arr.shape[0]):
		if arr[i,1]=='Accelerometer':
			time_accel.append(arr[i,0])
			accel=math.pow((float)(arr[i,2]),2)+math.pow((float)(arr[i,3]),2)+math.pow((float)(arr[i,4]),2)			
			acceleration.append(math.sqrt(accel)-9)
		elif arr[i,1]=='Linear Acceleration':
			time_accel.append(arr[i,0])
			accel=math.pow((float)(arr[i,2]),2)+math.pow((float)(arr[i,3]),2)+math.pow((float)(arr[i,4]),2)			
			acceleration.append(math.sqrt(accel))
		elif arr[i,1]=='Gyroscope':
			time_rotate.append(arr[i,0])
			rotate=math.pow((float)(arr[i,2]),2)+math.pow((float)(arr[i,3]),2)+math.pow((float)(arr[i,4]),2)
			rotation.append(math.sqrt(rotate))
		else :
			time_pedo.append(arr[i,0])
	acceleration=eliminate_noise(acceleration)
	rotation=eliminate_noise(rotation)
	return np.array(acceleration),np.array(rotation),np.array(time_accel),np.array(time_rotate),np.array(time_pedo)

def split_training_testing():
	for i in range(10):
		print((int)(random.random()*10))

def eliminate_noise(input_list):
	return_list=[input_list[0]]
	for i in range(1,len(input_list)-1):
		return_list.append((input_list[i-1]+input_list[i]+input_list[i+1])/3)
	return_list.append(input_list[-1])
	print("original length"+str(len(input_list))+" ;and the return list "+str(len(return_list)))
	return return_list

def step_divide_accel(training_or_testing,pedo_time,accel,accel_time):
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