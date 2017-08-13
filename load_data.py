import numpy as np

save_directory='feature_saved_directory'
tfeatures_bias_eli_path=save_directory+'/tfeature_bias_eli.npy'
sfeatures_bias_eli_path=save_directory+'/sfeature_bias_eli.npy'
tfeatures_std_path=save_directory+'/tfeature_standardised.npy'
sfeatures_std_path=save_directory+'/sfeature_standardised.npy'
slabel_arousal_path=save_directory+'/slabel_arousal.npy'
slabel_valence_path=save_directory+'/slabel_valence.npy'
tlabel_arousal_path=save_directory+'/tlabel_arousal.npy'
tlabel_valence_path=save_directory+'/tlabel_valence.npy'

def load(training_or_testing):
	if training_or_testing=='t':
		tfeatures_bias_eli=np.load(tfeatures_bias_eli_path)
		tfeatures_std=np.load(tfeatures_std_path)
		tlabel_valence=np.load(tlabel_valence_path)
		tlabel_arousal=np.load(tlabel_arousal_path)		
		return np.array([tfeatures_bias_eli,tfeatures_std]),[tlabel_arousal,tlabel_valence]
	else:		
		sfeatures_bias_eli=np.load(sfeatures_bias_eli_path)
		sfeatures_std=np.load(sfeatures_std_path)
		slabel_valence=np.load(slabel_valence_path)
		slabel_arousal=np.load(slabel_arousal_path)
		return np.array([sfeatures_bias_eli,sfeatures_std]),[slabel_arousal,slabel_valence]
def load_unstandardised(training_or_testing):
	if training_or_testing=='t':
		tfeature=np.load('feature_saved_directory/tfeature_raw.npy')
		tlabel_valence=np.load(tlabel_valence_path)
		tlabel_arousal=np.load(tlabel_arousal_path)
		return tfeature,[tlabel_arousal,tlabel_valence]
	else:		
		sfeature=np.load('feature_saved_directory/sfeature_raw.npy')
		slabel_valence=np.load(slabel_valence_path)
		slabel_arousal=np.load(slabel_arousal_path)
		return sfeature,[slabel_arousal,slabel_valence]