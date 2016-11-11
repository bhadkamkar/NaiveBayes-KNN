import numpy as np
import operator

#data hardcode
feature_count = 9
total_class_count = 7
INF = float("inf")
def NBgetMean(train_x,train_y):
	train_total_samples = train_y.size
	train_class_count = dict.fromkeys((i for i in range(1,total_class_count+1)),0)
	train_feature_sum = dict.fromkeys((i for i in range(1,total_class_count+1)),np.zeros(feature_count+1))
	train_class_mean = dict.fromkeys((i for i in range(1,total_class_count+1)),np.zeros(feature_count+1))
	for i in range(0,train_total_samples):
		sample_class = train_y[i]
		train_class_count[sample_class] += 1
		train_feature_sum[sample_class] = np.add(train_feature_sum[sample_class],train_x[i])
	for i in range(1,total_class_count+1):
		if train_class_count[i] != 0.0 :
			train_class_mean[i] = train_feature_sum[i]/train_class_count[i]
		else:
			train_class_mean[i].fill(0)
	return train_class_mean

def NBgetVariance(train_x,train_y,train_class_mean):
	train_total_samples = train_y.size
	train_class_count = dict.fromkeys((i for i in range(1,total_class_count+1)),0)
	train_class_variance_not_divided = dict.fromkeys((i for i in range(1,total_class_count+1)),np.zeros(feature_count+1))
	train_class_variance = dict.fromkeys((i for i in range(1,total_class_count+1)),np.zeros(feature_count+1))
	for i in range(0,train_total_samples):
		sample_class = train_y[i]
		train_class_count[sample_class] += 1
		train_class_variance_not_divided[sample_class] = np.add(train_class_variance_not_divided[sample_class],np.square(np.subtract(train_x[i],train_class_mean[sample_class])))
	for i in range(1,total_class_count+1):
		if train_class_count[i] != 0.0 :
			train_class_variance[i] = train_class_variance_not_divided[i]/train_class_count[i]
			#low_value_indices = (train_class_variance[i] == 0)
			#train_class_variance[i][low_value_indices] = INF
		else:
			train_class_variance[i].fill(INF)
	return train_class_variance

def NBgetClassProbabilities(train_y):
	train_class_count = dict.fromkeys((i for i in range(1,total_class_count+1)),0.0)
	train_class_log_probability = dict.fromkeys((i for i in range(1,total_class_count+1)),0.0)
	train_total_samples = train_y.size
	for i in range(0,train_total_samples):
		sample_class = train_y[i]
		train_class_count[sample_class] += 1.0
	for i in range(1,total_class_count+1):
		train_class_log_probability[i] = np.log(((train_class_count[i]+1)/(train_total_samples+1)))
	return train_class_log_probability
def NBpredictClass(sample_x,trained_params):
	class_score = dict.fromkeys((i for i in range(1,total_class_count+1)),0)
	for class_idx in range(1,total_class_count+1):
		term1 = 2*trained_params["logProb"][class_idx]		
		term2a = (np.square(np.subtract(sample_x,trained_params["mu"][class_idx]))[1:])
		term2b = trained_params["sigma^2"][class_idx][1:]
		zero_pos_2a1 = np.where(term2a == 0)[0]
		zero_pos_2a2 = np.where(term2a != 0)[0]
		zero_pos_2b = np.where(term2b == 0)[0]
		zero_pos_2c1 = np.intersect1d(zero_pos_2a1,zero_pos_2b)
		zero_pos_2c2 = np.intersect1d(zero_pos_2a2,zero_pos_2b)
		term3a = trained_params["sigma^2"][class_idx][1:]
		for pos in zero_pos_2c1:
			term2a[pos] = 0
			term2b[pos] = 1
			term3a[pos] = 1
		for pos in zero_pos_2c2:
			term2a[pos] = INF
			term2b[pos] = 1
			term3a[pos] = 1
		term2c = np.divide(term2a,term2b)
		term2 = np.sum(term2c)
		#term3a = trained_params["sigma^2"][class_idx][1:]
		term3b = np.log(term3a)
		term3 = np.sum(term3b)
		class_score[class_idx] = term1 - term2 - term3
		#print class_idx,class_score[class_idx]
		#print trained_params["mu"][class_idx]
		#print trained_params["sigma^2"][class_idx]
	return max(class_score.iteritems(), key=operator.itemgetter(1))[0]
	
def naiveBayes():
	train_x = np.loadtxt("train.txt",delimiter = ',',usecols = (i for i in range(0,feature_count+1)))
	train_y = np.loadtxt("train.txt",delimiter = ',',usecols = (feature_count+1,))
	train_class_mean = NBgetMean(train_x,train_y)
	train_class_variance = NBgetVariance(train_x,train_y,train_class_mean)
	train_class_log_probability = NBgetClassProbabilities(train_y)
	trained_params = {
				"mu":train_class_mean,
				"sigma^2":train_class_variance,
				"logProb":train_class_log_probability
			}
	print "******Naive bayes******"
	correct_count_train = 0.0
	#print "Predicting on Training set"
	#print "SR.NO.\tPredicted class\tActual class"
	for i in range(0,train_y.size):
		sample_x = train_x[i]
		predict_y = NBpredictClass(sample_x,trained_params)
		#print i,"\t",predict_y,"\t\t",int(train_y[i])
		if(predict_y == train_y[i]):
			correct_count_train += 1
	print "Train accuracy is",correct_count_train/train_y.size

	correct_count_test = 0.0
	test_x = np.loadtxt("test.txt",delimiter = ',',usecols = (i for i in range(0,feature_count+1)))
	test_y = np.loadtxt("test.txt",delimiter = ',',usecols = (feature_count+1,))
	#print "Predicting on Testing set"
	#print "SR.NO.\tPredicted class\tActual class"
	for i in range(0,test_y.size):
		sample_x = test_x[i]
		predict_y = NBpredictClass(sample_x,trained_params)
		#print i,"\t",predict_y,"\t\t",int(test_y[i])
		if(predict_y == test_y[i]):
			#print test_x[i][0],test_y[i]
			correct_count_test += 1
	
	print "Test accuracy is",correct_count_test/test_y.size

class data:
	def __init__ (self,dist,label):
		self.dist = dist
		self.label = label
	def setParams(self,dist,label):
		self.dist = dist
		self.label = label
	
def knn():
	print "******K nearest neighbors******"
	train_x = np.loadtxt("train.txt",delimiter = ',',usecols = (i for i in range(1,feature_count+1)))
	train_y = np.loadtxt("train.txt",delimiter = ',',usecols = (feature_count+1,))
	mean = np.divide(np.sum(train_x,axis=0),train_y.size)
	sigma = np.divide(np.sqrt(np.sum(np.square(np.subtract(train_x,mean)),axis = 0)),(train_y.size-1))
	train_x_norm = np.divide(np.subtract(train_x,mean),sigma)
	
	test_x = np.loadtxt("test.txt",delimiter = ',',usecols = (i for i in range(1,feature_count+1)))
	test_y = np.loadtxt("test.txt",delimiter = ',',usecols = (feature_count+1,))
	test_x_norm = np.divide(np.subtract(test_x,mean),sigma)

	L1Train = [[data(0,0) for i in range(0,train_y.size)] for j in range(0,train_y.size)]
	L2Train = [[data(0,0) for i in range(0,train_y.size)] for j in range(0,train_y.size)]
	L1Test = [[data(0,0) for i in range(0,train_y.size)] for j in range(0,test_y.size)]
	L2Test = [[data(0,0) for i in range(0,train_y.size)] for j in range(0,test_y.size)]
	
	trainL1class_count_k1 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL1class_count_k3 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL1class_count_k5 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL1class_count_k7 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL2class_count_k1 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL2class_count_k3 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL2class_count_k5 = np.zeros((train_y.size,total_class_count+1),dtype=int)
	trainL2class_count_k7 = np.zeros((train_y.size,total_class_count+1),dtype=int)

	
	testL1class_count_k1 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL1class_count_k3 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL1class_count_k5 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL1class_count_k7 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL2class_count_k1 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL2class_count_k3 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL2class_count_k5 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	testL2class_count_k7 = np.zeros((test_y.size,total_class_count+1),dtype=int)
	
	for a in range(0,train_y.size):
		for b in range(0,train_y.size):
			label = train_y[b]
			distL1 = np.sum(np.abs(np.subtract(train_x_norm[a],train_x_norm[b])))
			if distL1 == 0:
				distL1 = INF 
			distL2 = np.sqrt(np.sum(np.square(np.subtract(train_x_norm[a],train_x_norm[b]))))
			if distL2 == 0:
				distL2 = INF 
			L1Train[a][b].dist = distL1
			L1Train[a][b].label = int(label)			
			L2Train[a][b].dist = distL2
			L2Train[a][b].label = int(label)
	for a in range(0,train_y.size):
		L1Train[a].sort(key=lambda x: x.dist)
		L2Train[a].sort(key=lambda x: x.dist)

	for a in range(0,test_y.size):
		for b in range(0,train_y.size):
			label = train_y[b]
			distL1 = np.sum(np.abs(np.subtract(test_x_norm[a],train_x_norm[b])))
			 
			distL2 = np.sqrt(np.sum(np.square(np.subtract(test_x_norm[a],train_x_norm[b]))))
			
			L1Test[a][b].dist = distL1
			L1Test[a][b].label = int(label)			
			L2Test[a][b].dist = distL2
			L2Test[a][b].label = int(label)
	for a in range(0,test_y.size):
		L1Test[a].sort(key=lambda x: x.dist)
		L2Test[a].sort(key=lambda x: x.dist)
	#train
	for a in range(0,train_y.size):
		for b in range(0,1):
			trainL1class_count_k1[a][L1Train[a][b].label] += 1
		for b in range(0,3):
			trainL1class_count_k3[a][L1Train[a][b].label] += 1
		for b in range(0,5):
			trainL1class_count_k5[a][L1Train[a][b].label] += 1
		for b in range(0,7):
			trainL1class_count_k7[a][L1Train[a][b].label] += 1

		for b in range(0,1):
			trainL2class_count_k1[a][L2Train[a][b].label] += 1
		for b in range(0,3):
			trainL2class_count_k3[a][L2Train[a][b].label] += 1
		for b in range(0,5):
			trainL2class_count_k5[a][L2Train[a][b].label] += 1
		for b in range(0,7):
			trainL2class_count_k7[a][L2Train[a][b].label] += 1

	trainL1k1PredictCorrect=0
	trainL1k3PredictCorrect=0
	trainL1k5PredictCorrect=0
	trainL1k7PredictCorrect=0
	trainL2k1PredictCorrect=0
	trainL2k3PredictCorrect=0
	trainL2k5PredictCorrect=0
	trainL2k7PredictCorrect=0		
	for a in range(0,train_y.size):
		if int(train_y[a])==np.argmax(trainL1class_count_k1[a]):
			trainL1k1PredictCorrect += 1
		keyListL1k3 = []
		k = np.argmax(trainL1class_count_k3[a])
		for key in range(1,total_class_count+1):
			if trainL1class_count_k3[a][key] == trainL1class_count_k3[a][k]:
				keyListL1k3 = keyListL1k3 + [key]
		for b in range(0,3):
			if  L1Train[a][b].label	in keyListL1k3:
				if int(train_y[a]) == L1Train[a][b].label:
					trainL1k3PredictCorrect += 1
				break
		keyListL1k5 = []
		k = np.argmax(trainL1class_count_k5[a])
		for key in range(1,total_class_count+1):
			if trainL1class_count_k5[a][key] == trainL1class_count_k5[a][k]:
				keyListL1k5 = keyListL1k5 + [key]

		for b in range(0,5):
			if  L1Train[a][b].label	in keyListL1k5:
				if int(train_y[a]) == L1Train[a][b].label:
					trainL1k5PredictCorrect += 1
				break

		keyListL1k7 = []
		k = np.argmax(trainL1class_count_k7[a])
		for key in range(1,total_class_count+1):
			if trainL1class_count_k7[a][key] == trainL1class_count_k7[a][k]:
				keyListL1k7 = keyListL1k7 + [key]
		for b in range(0,7):
			if  L1Train[a][b].label	in keyListL1k7:
				if int(train_y[a]) == L1Train[a][b].label:
					trainL1k7PredictCorrect += 1
				break
		
		if int(train_y[a])==np.argmax(trainL2class_count_k1[a]):
			trainL2k1PredictCorrect += 1
		keyListL2k3 = []
		k = np.argmax(trainL2class_count_k3[a])
		for key in range(1,total_class_count+1):
			if trainL2class_count_k3[a][key] == trainL2class_count_k3[a][k]:
				keyListL2k3 = keyListL2k3 + [key]
		for b in range(0,3):
			if  L2Train[a][b].label	in keyListL2k3:
				if int(train_y[a]) == L2Train[a][b].label:
					trainL2k3PredictCorrect += 1
				break
		keyListL2k5 = []
		k = np.argmax(trainL2class_count_k5[a])
		for key in range(1,total_class_count+1):
			if trainL2class_count_k5[a][key] == trainL2class_count_k5[a][k]:
				keyListL2k5 = keyListL2k5 + [key]

		for b in range(0,5):
			if  L2Train[a][b].label	in keyListL2k5:
				if int(train_y[a]) == L2Train[a][b].label:
					trainL2k5PredictCorrect += 1
				break

		keyListL2k7 = []
		k = np.argmax(trainL2class_count_k7[a])
		for key in range(1,total_class_count+1):
			if trainL2class_count_k7[a][key] == trainL2class_count_k7[a][k]:
				keyListL2k7 = keyListL2k7 + [key]
		for b in range(0,7):
			if  L2Train[a][b].label	in keyListL2k7:
				if int(train_y[a]) == L2Train[a][b].label:
					trainL2k7PredictCorrect += 1
				break
		
	print "Train L1 K=1 accruacy is", float(trainL1k1PredictCorrect)/train_y.size
	print "Train L1 K=3 accruacy is", float(trainL1k3PredictCorrect)/train_y.size
	print "Train L1 K=5 accruacy is", float(trainL1k5PredictCorrect)/train_y.size
	print "Train L1 K=7 accruacy is", float(trainL1k7PredictCorrect)/train_y.size
	print "Train L2 K=1 accruacy is", float(trainL2k1PredictCorrect)/train_y.size
	print "Train L2 K=3 accruacy is", float(trainL2k3PredictCorrect)/train_y.size
	print "Train L2 K=5 accruacy is", float(trainL2k5PredictCorrect)/train_y.size
	print "Train L2 K=7 accruacy is", float(trainL2k7PredictCorrect)/train_y.size
	

	#test
	for a in range(0,test_y.size):
		for b in range(0,1):
			testL1class_count_k1[a][L1Test[a][b].label] += 1
		for b in range(0,3):
			testL1class_count_k3[a][L1Test[a][b].label] += 1
		for b in range(0,5):
			testL1class_count_k5[a][L1Test[a][b].label] += 1
		for b in range(0,7):
			testL1class_count_k7[a][L1Test[a][b].label] += 1

		for b in range(0,1):
			testL2class_count_k1[a][L2Test[a][b].label] += 1
		for b in range(0,3):
			testL2class_count_k3[a][L2Test[a][b].label] += 1
		for b in range(0,5):
			testL2class_count_k5[a][L2Test[a][b].label] += 1
		for b in range(0,7):
			testL2class_count_k7[a][L2Test[a][b].label] += 1

	testL1k1PredictCorrect=0
	testL1k3PredictCorrect=0
	testL1k5PredictCorrect=0
	testL1k7PredictCorrect=0
	testL2k1PredictCorrect=0
	testL2k3PredictCorrect=0
	testL2k5PredictCorrect=0
	testL2k7PredictCorrect=0		
	for a in range(0,test_y.size):
		if int(test_y[a])==np.argmax(testL1class_count_k1[a]):
			testL1k1PredictCorrect += 1
		keyListL1k3 = []
		k = np.argmax(testL1class_count_k3[a])
		for key in range(1,total_class_count+1):
			if testL1class_count_k3[a][key] == testL1class_count_k3[a][k]:
				keyListL1k3 = keyListL1k3 + [key]
		for b in range(0,3):
			if  L1Test[a][b].label	in keyListL1k3:
				if int(test_y[a]) == L1Test[a][b].label:
					testL1k3PredictCorrect += 1
				break
		keyListL1k5 = []
		k = np.argmax(testL1class_count_k5[a])
		for key in range(1,total_class_count+1):
			if testL1class_count_k5[a][key] == testL1class_count_k5[a][k]:
				keyListL1k5 = keyListL1k5 + [key]

		for b in range(0,5):
			if  L1Test[a][b].label	in keyListL1k5:
				if int(test_y[a]) == L1Test[a][b].label:
					testL1k5PredictCorrect += 1
				break

		keyListL1k7 = []
		k = np.argmax(testL1class_count_k7[a])
		for key in range(1,total_class_count+1):
			if testL1class_count_k7[a][key] == testL1class_count_k7[a][k]:
				keyListL1k7 = keyListL1k7 + [key]
		for b in range(0,7):
			if  L1Test[a][b].label	in keyListL1k7:
				if int(test_y[a]) == L1Test[a][b].label:
					testL1k7PredictCorrect += 1
				break
		
		if int(test_y[a])==np.argmax(testL2class_count_k1[a]):
			testL2k1PredictCorrect += 1
		keyListL2k3 = []
		k = np.argmax(testL2class_count_k3[a])
		for key in range(1,total_class_count+1):
			if testL2class_count_k3[a][key] == testL2class_count_k3[a][k]:
				keyListL2k3 = keyListL2k3 + [key]
		for b in range(0,3):
			if  L2Test[a][b].label	in keyListL2k3:
				if int(test_y[a]) == L2Test[a][b].label:
					testL2k3PredictCorrect += 1
				break
		keyListL2k5 = []
		k = np.argmax(testL2class_count_k5[a])
		for key in range(1,total_class_count+1):
			if testL2class_count_k5[a][key] == testL2class_count_k5[a][k]:
				keyListL2k5 = keyListL2k5 + [key]

		for b in range(0,5):
			if  L2Test[a][b].label	in keyListL2k5:
				if int(test_y[a]) == L2Test[a][b].label:
					testL2k5PredictCorrect += 1
				break

		keyListL2k7 = []
		k = np.argmax(testL2class_count_k7[a])
		for key in range(1,total_class_count+1):
			if testL2class_count_k7[a][key] == testL2class_count_k7[a][k]:
				keyListL2k7 = keyListL2k7 + [key]
		for b in range(0,7):
			if  L2Test[a][b].label	in keyListL2k7:
				if int(test_y[a]) == L2Test[a][b].label:
					testL2k7PredictCorrect += 1
				break
		
	print "Test L1 K=1 accruacy is", float(testL1k1PredictCorrect)/test_y.size
	print "Test L1 K=3 accruacy is", float(testL1k3PredictCorrect)/test_y.size
	print "Test L1 K=5 accruacy is", float(testL1k5PredictCorrect)/test_y.size
	print "Test L1 K=7 accruacy is", float(testL1k7PredictCorrect)/test_y.size
	print "Test L2 K=1 accruacy is", float(testL2k1PredictCorrect)/test_y.size
	print "Test L2 K=3 accruacy is", float(testL2k3PredictCorrect)/test_y.size
	print "Test L2 K=5 accruacy is", float(testL2k5PredictCorrect)/test_y.size
	print "Test L2 K=7 accruacy is", float(testL2k7PredictCorrect)/test_y.size

	
	
naiveBayes()
knn()

