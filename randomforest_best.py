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
	with open('result/random_forest_best','a') as f:
		f.write('\n\n'+str(datetime.datetime.now())+'\n\n')
		f.write('no_bias_elimination' if std_or_bias=='std' else 'with_bias_elimination')	
		for i in range(2):
			ttarget=tlabels[i][1]                               # the number projection of the two dimensions
			starget=slabels[i][1]
			max_accuracy=[]
			n_estimators=[]
			predictions=[]
			for j in range(10,100,5):				            # try to find the best parameter
				clf = RandomForestClassifier(n_estimators=j)	# parameter has to be adjusted
				clf.fit(tfeature[i],ttarget)
				accuracy=clf.score(sfeature,starget)
				max_accuracy.append(accuracy)
				n_estimators.append(j)
				predictions.append(clf.predict(sfeature))
			max_accuracy=np.array(max_accuracy)
			best_accuracy=np.max(max_accuracy)                  # the highest accuracy
			best_estimator=n_estimators[np.argmax(max_accuracy)]# the number of estimators used to achieve
			best_prediction=predictions[np.argmax(max_accuracy)]
			#clf = RandomForestClassifier(n_estimators=best_estimator)
			#clf.fit(tfeature[i],ttarget)
			#prediction=clf.predict(sfeature)
			#print(prediction)
			#prob_matrix=clf.predict_proba(sfeature)
			projected_target=np.array(['middle' for _ in range(len(best_prediction))])
			for z in range(len(best_prediction)):
				if int(best_prediction[z])==0:
					projected_target[z]='low'
				elif int(best_prediction[z])==2:
					projected_target[z]='high'
				else:
					pass
			print(projected_target)
			confusion_table=pd.crosstab(slabels[i][0],projected_target,rownames=['Actual degree'],colnames=['Predicted degree'])
			print(confusion_table)
			if i==0:
				f.write('\n\n\tarousal:'+'\n')
				# arousal
			else:
				f.write('\n\n\tvalence:'+'\n')
			f.write('\t\t\toptimal n_estimator '+str(best_estimator))
			f.write('\t\toptimal accuracy '+str(best_accuracy)+'\n\n')
			f.write(str(confusion_table))
		f.close()
random_forest('bias')
random_forest('std')