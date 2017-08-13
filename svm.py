import numpy as np
import pandas as pd
from sklearn.svm import SVC
from load_data import load
import datetime

tfeatures,tlabels=load('t')								     # tfeatures[0] is eli_biased;tfeatures[1] is std
sfeatures,slabels=load('s')									 # tlabls[0] is arousal tlabels[0][0] is text.tlabels[0][1] is number

def svm(std_or_bias_eli='std'):
	if(std_or_bias_eli=='std'):
		tfeature=tfeatures[1]
		sfeature=sfeatures[1]
	else:
		tfeature=tfeatures[0]
		sfeature=sfeatures[0]		
	with open('result/svm_standardised','a') as f:
		f.write('\n\n'+str(datetime.datetime.now())+'\n\n')
		f.write('standardised\n\n' if std_or_bias_eli=='std' else 'personal_biased_eliminated\n\n')	
		for i in range(2):
			if i==0:
				f.write('\n\n\tarousal:'+'\n')
			else:
				f.write('\n\n\tvalence:'+'\n')
			ttarget=tlabels[i][1]                               # the number projection of the two dimensions
			starget=slabels[i][1]			
			clf = SVC(kernel='poly',degree=4,coef0=4,gamma=0.4,decision_function_shape='ovr')	# parameter has to be adjusted
			clf.fit(tfeature[i],ttarget)
			accuracy=clf.score(sfeature,starget)
			confusion_table=pd.crosstab(slabels[i][1],clf.predict(sfeature),rownames=['Actual degree'],colnames=['Predicted degree'])
			#f.write('\t\t\tgamma '+str(j))
			f.write('\t\taccuracy '+str(accuracy)+'\n\n')
			f.write(str(confusion_table))
			f.write('\n\n')			
	f.close()
svm('std')
svm('bias')
