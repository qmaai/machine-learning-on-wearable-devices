import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

save_directory='feature_saved_directory'
tfeatures_raw_path=save_directory+'/tfeature_raw.npy'
sfeatures_raw_path=save_directory+'/sfeature_raw.npy'
tfeatures_bias_eli_path=save_directory+'/tfeature_bias_eli'
sfeatures_bias_eli_path=save_directory+'/sfeature_bias_eli'
tfeatures_std_path=save_directory+'/tfeature_standardised'
sfeatures_std_path=save_directory+'/sfeature_standardised'
slabel_arousal_path=save_directory+'/slabel_arousal'
slabel_valence_path=save_directory+'/slabel_valence'
tlabel_arousal_path=save_directory+'/tlabel_arousal'
tlabel_valence_path=save_directory+'/tlabel_valence'

tlabel_raw=pd.read_csv('tlabel.csv')
tlabel_raw=tlabel_raw.as_matrix()
tlabel_raw=tlabel_raw[~np.isnan(tlabel_raw)]
tlabel_arousal=tlabel_raw[0:len(tlabel_raw):2]
tlabel_valence=tlabel_raw[1:len(tlabel_raw):2]

slabel_raw=pd.read_csv('slabel.csv')
slabel_raw=slabel_raw.as_matrix()
slabel_raw=slabel_raw[~np.isnan(slabel_raw)]
slabel_arousal=slabel_raw[0:len(slabel_raw):2]
slabel_valence=slabel_raw[1:len(slabel_raw):2]


counter=0
for i in [slabel_arousal,slabel_valence,tlabel_arousal,tlabel_valence]:
	label=np.array(['middle' for _ in range(len(i))])
	label[np.argwhere(i>1)]='high'
	label[np.argwhere(i<-1)]='low'
	label_num=np.array([1 for _ in range(len(i))])
	label_num[np.argwhere(i>1)]=2
	label_num[np.argwhere(i<-1)]=0
	if counter==0:
		np.save(slabel_arousal_path,np.array([label,label_num]))
		print('slabel_arousal')
		print(label)
	elif counter==1:
		np.save(slabel_valence_path,np.array([label,label_num]))
		print('slabel_valence')
		print(label)
	elif counter==2:
		np.save(tlabel_arousal_path,np.array([label,label_num]))
		print('tlabel_arousal')
		print(label)
	else:
		np.save(tlabel_valence_path,np.array([label,label_num]))
		print('tlabel_valence')
		print(label)
	counter+=1

tfeatures_raw=np.load(tfeatures_raw_path)
sfeatures_raw=np.load(sfeatures_raw_path)

tfeatures_std=np.copy(tfeatures_raw)
sfeatures_std=np.copy(sfeatures_raw)
np.save(tfeatures_std_path,tfeatures_std)
np.save(sfeatures_std_path,sfeatures_std)


## personal bias elimination
for i in (0,127,8):
	for j in range(len(tfeatures_raw[0])):
		tfeatures_raw[i:i+8,j]=tfeatures_raw[i:i+8,j]-np.mean(tfeatures_raw[i:i+8,j])
for i in (128,135,7):
	for j in range(len(tfeatures_raw[0])):
		tfeatures_raw[i:i+7,j]=tfeatures_raw[i:i+7,j]-np.mean(tfeatures_raw[i:i+7,j])
for i in range(135,len(tfeatures_raw),8):
	for j in range(len(tfeatures_raw[0])):
		tfeatures_raw[i:i+8,j]=tfeatures_raw[i:i+8,j]-np.mean(tfeatures_raw[i:i+8,j])
for i in range(0,len(sfeatures_raw[0]),2):
	for j in range(len(sfeatures_raw)):
		sfeatures_raw[i:i+2,j]=sfeatures_raw[i:i+2,j]-np.mean(sfeatures_raw[i:i+2,j])

sc=StandardScaler()
tfeatures_bias_eli=sc.fit_transform(tfeatures_raw)
sfeatures_bias_eli=sc.fit_transform(sfeatures_raw)
np.save(tfeatures_bias_eli_path,tfeatures_bias_eli)
np.save(sfeatures_bias_eli_path,sfeatures_bias_eli)


'''
print(tfeatures_bias_eli.shape)
print(sfeatures_bias_eli.shape)
print(tfeatures_bias_eli[0])
print(sfeatures_bias_eli[0])
print(np.argwhere(np.isnan(tfeatures_raw)))
print(np.argwhere(np.isnan(sfeatures_raw)))
'''