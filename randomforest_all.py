import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from load_data import load
import datetime

tfeatures,tlabels=load('t')								     # tfeatures[0] is eli_biased;tfeatures[1] is std
sfeatures,slabels=load('s')									 # tlabls[0] is arousal tlabels[0][0] is text.tlabels[0][1] is number

def random_forest(std_or_bias):
	if std_or_bias=='std':
		tfeature=tfeatures[1]
		sfeature=sfeatures[1]
	else:
		tfeature=tfeatures[0]
		sfeature=sfeatures[0]
	with open('result/random_forest_all','a') as f:
		f.write('\n\n'+str(datetime.datetime.now())+'\n\n')
		f.write('no_bias_elimination' if std_or_bias=='std' else 'with_bias_eli')	
		for i in range(2):
			if i==0:
				f.write('\n\n\tarousal:'+'\n')
			else:
				f.write('\n\n\tvalence:'+'\n')
			ttarget=tlabels[i][1]                               # the number projection of the two dimensions
			starget=slabels[i][1]
			for j in range(10,100,5):				            # try to find the best parameter
				clf = RandomForestClassifier(n_estimators=j)	# parameter has to be adjusted
				clf.fit(tfeature[i],ttarget)
				prediction=clf.predict(sfeature)
				accuracy=clf.score(sfeature,starget)
				confusion_table=pd.crosstab(slabels[i][1],prediction,rownames=['Actual degree'],colnames=['Predicted degree'])
				f.write('\t\t\tn_estimator '+str(j))
				f.write('\t\taccuracy '+str(accuracy)+'\n\n')
				f.write(str(confusion_table))
				f.write('\n\n')				
	f.close()
random_forest('bias')
random_forest('std')