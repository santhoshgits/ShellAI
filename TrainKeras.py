import time
import pandas
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.layers import LSTM, concatenate, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Dropout, RepeatVector, TimeDistributed, Dropout, Reshape, MaxPooling3D, RNN
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras import layers
import os
from collections import Counter, defaultdict
import random
from PIL import Image
import sys		
import copy
from matplotlib.image import imread
from scipy import misc
from scipy import stats
import math
import tensorflow_addons as tfa

dire = os.getcwd()

CutOff = 0


aline = open(dire+'/train/train_mod.csv').readlines()
X_Total = []
Y_Total = []
for line in aline[1:]:
	line = line.strip()
	l = line.split(',')
	if float(l[9]) >= CutOff:
		X_Total.append(l[2:])
		Y_Total.append(l[9])
		
X_Total = np.asarray(X_Total, dtype='float')		
Y_Total = np.asarray(Y_Total, dtype='float')
		
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_Total)


Y_Total = np.reshape(Y_Total, (-1, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
scalerY.fit(Y_Total)



TimeStampDict_temp = defaultdict(list)
for line in aline[1:]:
	line = line.strip()
	l = line.split(',')
	#arr = []
	#print(l)
	if float(l[9]) >= CutOff:
		TimeStampDict_temp[l[0]].append(l[1:])
	


def formatDate(val):
	val = val.split('/')
	a, b = 0, 0
	if len(val[0]) == 1:
		a = '0'+val[0]
	else:
		a = val[0]
	if len(val[1]) == 1:
		b = '0'+val[1]
	else:
		b = val[1]
	return a+b		



TimeSeriesImage = defaultdict(list)
for i,j in TimeStampDict_temp.items():
	#print (i)
	arr = [ formatDate(i)+''.join(k[0].split(':'))+'00' for k in j ]
	for k in range(len(arr)):
		#print(arr[k-3:k+3])
		for l in arr[k-3:k+3]:
			TimeSeriesImage[arr[k]].append(l)



def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(1):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def get_splits(arr):
	arr1 = []
	for i in range(0, len(arr)):
		arr1.append(arr[i])
	return arr1
	



def do_difference(arr):
	arr1 = []
	for i in range(1,len(arr)):
		'''
		val1 = arr[i-1][1:]
		val2 = arr[i][1:]
		val1 = np.asarray(val1, dtype='float')
		val2 = np.asarray(val2, dtype='float')
		#arr1.append(val2-val1)
		'''
		val2 = arr[i]
		arr1.append(val2)
	return arr1
	
	
def get_series(seqs):
	n = random.randint(1,5)
	arr = []
	for i in range(n,len(seqs),5):
		arr.append(seqs[i])
	return seqs
















def GenInputs(seqs, dirs, N, Step,R, Break):
	#print(N, Step, N+Step)
	
	steps = N + Step
	#dirs = '0101'
	#print(N, steps, Step)
	
	
	seqs = get_series(seqs)
	#print(len(seqs),'---')
	ranges = (list(range(Step,len(seqs)-N)))
	#print(ranges)
	
	random.shuffle(ranges)
	
	Rint = []
	
	
	dic = {}
	for i in ranges:
		if i not in dic:
			Rint.append(i)
		#for j in range(i-1,i+1):
		#	dic[j] = 0
	
	
	X, Y, I = [], [], []
	
	for t in Rint:
	
	
		LR = t
		LArr = seqs[LR-Step:LR]
		file_loc = []
		#print(len(LArr), Step, LR)
		
		dire1 = dire+'/train/'+dirs
		dic = { i:0 for i in os.listdir(dire1) if 'sant' in i }
		#print(dic)
		
		dic_track = {}
		for i in LArr[-300:]:
			h, m = i[0].split(':')
			#print(dirs+h+m)
			if dirs+h+m+'00' in TimeSeriesImage:
				#print('enter1')
				for k in TimeSeriesImage[dirs+h+m+'00']:
					files = k+'_sant.jpg'
					if files in dic:
						if files not in dic_track:
							dic_track[files] = 0
							file_loc.append(files)
					
			
		if len(file_loc) >= 6:
			
			imaged = []
			#print(dire1)
			#print(file_loc[-6:])
			for i in file_loc[-6:]:
				image = Image.open(dire1+'/'+i)
				image = image.resize((32,32))
				image = np.asarray(image)
				image = normalize(image)
				image = np.reshape(image, (32,32,1))
				imaged.append(image)
				
			imaged = np.asarray(imaged)
			
			#print(imaged.shape)
			#imaged = np.concatenate(imaged, axis=0)
			#print(imaged.shape)
			#time.sleep(11)
			x = [ i[1:] for i in LArr ]
			x = x[-Step:]
			x = scaler.transform(x)
			
			y = [ i[8] for i in seqs[LR:LR+N] ]
			
			y = np.asarray(y, dtype='float')
			y = np.reshape(y, (-1, 1))
			y = scalerY.transform(y)
			y = y.flatten()
			y = y[-1:]
			#y = y[-1]
			
			#print(x.shape)
			#time.sleep(11)
			
			imaged = np.reshape(imaged, (6, 32,32, 1) )
			#imaged = np.zeros((1, 6, 32, 32, 1))
			y = np.reshape(y, 1)
			
			return x, y, imaged, True
	
	return None, None, None, False
			























def PrepareTestData(N, S, files):
	Dir = os.listdir(dire+'/test/'+files)
	dire1 = dire+'/test/'+files
	Dir = [ j for j in Dir if 'sant.jpg' in j ]
	
	aline = open(dire+'/test/'+files+'/weather_data1.csv', 'r').readlines()
	aline = aline[1:]
	seqs = [ line.strip().split(',') for line in aline if float(line.split(',')[8]) >= CutOff ]
	
	seqs = seqs[::-1]
	Seqs = []
	for i in range(0, len(seqs)):
		Seqs.append(seqs[i])
	seqs = Seqs[::-1]
	
	
	N = N
	Step = S
	steps = N + Step
	#print(N, steps, S)
	
	
	
	
	seqs = get_series(seqs)
	#print(len(seqs),'---')
	ranges = (list(range(Step,len(seqs)-N)))
	#print(ranges)
	
	random.shuffle(ranges)
	
	Rint = []
	
	
	dic = {}
	for i in ranges:
		if i not in dic:
			Rint.append(i)
		#for j in range(i-1,i+1):
		#	dic[j] = 0
	
	
	X, Y, I = [], [], []
	#print(Rint)
	for t in Rint:
	
	
		LR = t
		LArr = seqs[LR-Step:LR]
		file_loc = []
		
		
		dire1 = dire+'/test/'+files
		Dir = os.listdir(dire+'/test/'+files)
		Dir = [ j for j in Dir if 'sant.jpg' in j ]
		
		dic = { i:0 for i in os.listdir(dire1) if 'sant' in i }
		#print(dic)
		
		dic_track = {}
		for i in LArr[-300:]:
			#print(i)
			filed = i[0]+'_sant.jpg'
			if filed in dic:
				if filed not in dic_track:
					dic_track[filed] = 0
					file_loc.append(filed)
		#print(file_loc)		
			
		if len(file_loc) >= 6:
			imaged = []
			Dir = file_loc[-6:]
			for i in Dir:
				image = Image.open(dire+'/test/'+files+'/'+i)
				image = image.resize((32,32))

				image = np.asarray(image)
				image = normalize(image)
				image = np.reshape(image, (32, 32,1))
				imaged.append(image)
			imaged = np.asarray(imaged)	
			imaged = np.reshape(imaged, (1, 6, 32, 32, 1) )
					
			x = [ i[1:] for i in LArr ]
			x = x[-Step:]
			x = scaler.transform(x)
			
			y = [ i[8] for i in seqs[LR:LR+N] ]	
			y = np.asarray(y, dtype='float')
			y = np.reshape(y, (-1, 1))
			y = scalerY.transform(y)
			y = y.flatten()
			y = y[-1:]
			
			return x, y, imaged, True
			
	return None, None, None, False	
		
		
		
		
	




def PrepareTestDataMain(N, S, files):
	#files = '237'
	
	Dir = os.listdir(dire+'/test/'+files)
	dire1 = dire+'/test/'+files
	Dir = [ j for j in Dir if 'sant.jpg' in j ]
	
	aline = open(dire+'/test/'+files+'/weather_data1.csv', 'r').readlines()
	aline = aline[1:]
	seqs = [ line.strip().split(',') for line in aline if float(line.split(',')[8]) >= CutOff ]
	
	seqs = seqs[::-1]
	Seqs = []
	for i in range(0, len(seqs)):
		Seqs.append(seqs[i])
	#print(len(Seqs))	
	seqs = Seqs[::-1]
	seqs = seqs[-S:]
	seqs = [ i[1:] for i in seqs ]
	seqs = np.asarray(seqs, dtype='float')
	seqs = scaler.transform(seqs)
	#print(seqs.shape)
	
	Dir = sorted(Dir, key = lambda x:int(x.split('_sant.jpg')[0]))
	Dir = Dir[-6:]
	
	imaged = []
	for i in Dir:
		image = Image.open(dire+'/test/'+files+'/'+i)
		image = image.resize((32,32))
		image = np.asarray(image)
		image = normalize(image)
		image = np.reshape(image, (32, 32,1))
		imaged.append(image)
	imaged = np.asarray(imaged)
	imaged = np.reshape(imaged, (1, 6, 32, 32, 1) )
	#imaged = np.concatenate(imaged, axis=0)
	#imaged = np.reshape(imaged, (1, 192,32,1) )
	return seqs, imaged
	
	
		
		
	
	
	
	
TimeStampDict = {}
for i,j in TimeStampDict_temp.items():
	#print (i)
	#print(j)
	arr = get_splits(j)
	#arr = do_difference(arr)
	
	TimeStampDict[i] = arr
	

#for steps in [60,90,120,150]:
for steps in [80,100,110,130,140,150,160,170]:
	steps = 120
	step=steps

	#step = 150
	N = 120

	#print(N)
	Main = []


	i2 = list(TimeStampDict.keys())
	
	#print(len(TimeStampDict[i]))
	#sys.exit()

	
	def data_generators():
		X, Y, I = [], [], []
		while True:
			i = random.choice(i2)
			i1 = formatDate(i)
			#print(i1)
			seqs = TimeStampDict[i]
			x, y, image, check = GenInputs(seqs, i1, N, step, 5, False)
			#print(len(X), check)
			#if len(Y) >= 10:
			#	yield I, Y
			if check:
				#print (x.shape)
				x =np.reshape(x, (1,steps,15))
				yield [x, image], y
				
				
				
		
	#Step = step
	Step = steps
	
	
	
	for i in os.listdir(dire+'/test'):
		
		x, y, imaged, check = PrepareTestData(N, steps, i)
	
	
	
	inputs1 = tf.keras.Input(shape=(Step,15))
	
	
	
	lstm1 = LSTM(64, input_shape=(Step, 15), return_sequences=True)(inputs1)
	lstm1 = LSTM(64, input_shape=(Step, 15), return_sequences=True)(lstm1)
	lstm1 = LSTM(64, input_shape=(Step, 15), return_sequences=True)(lstm1)
	lstm1 = LSTM(32, input_shape=(Step, 15))(lstm1)
	lstm1 = Dense(32, activation='relu')(lstm1)
	lstm1 = Dense(32, activation='relu')(lstm1)
	#lstm1 = Dense(1)(lstm1)
	
	
	'''
	inputs1 = tf.keras.Input(shape=(Step,15))
	lstm1 = LSTM(200, activation='relu', input_shape=(Step, 15), return_sequences=True)(inputs1)
	lstm1 = LSTM(200, activation='relu', input_shape=(Step, 15))(lstm1)
	lstm1 = RepeatVector(N)(lstm1)
	lstm1 = LSTM(200, activation='relu', return_sequences=True)(lstm1)
	lstm1 = LSTM(200, activation='relu', return_sequences=True)(lstm1)
	lstm1 = TimeDistributed(Dense(100, activation='relu'))(lstm1)
	lstm1 = TimeDistributed(Dense(100, activation='relu'))(lstm1)
	lstm1 = TimeDistributed(Dense(1))(lstm1)
	lstm1 = Flatten()(lstm1)
	'''
	
	
	inputs2 = tf.keras.Input(shape=(6, 32, 32, 1))
	x = layers.ConvLSTM2D(
		filters=16,
		kernel_size=(2,2),
		
		padding="same",
		return_sequences=True,
		activation="relu"
	)(inputs2)
	#x = layers.BatchNormalization()(x)
	x = MaxPooling3D(pool_size=(1,4,4))(x)
	x = layers.ConvLSTM2D(
		filters=16,
		kernel_size=(2,2),
		
		padding="same",
		return_sequences=True,
		activation="relu",
	)(x)
	#x = layers.BatchNormalization()(x)
	x = MaxPooling3D(pool_size=(1,4,4))(x)
	
		
	lstm2 = Reshape((6,2,2,16))(x)
	#lstm2 = Reshape((6,18,18,1))(x)
	lstm2 = Flatten()(lstm2)
	lstm2 = Dense(4, activation='relu')(lstm2)
	lstm2 = Dense(4, activation='relu')(lstm2)
	#lstm2 = Dense(1)(lstm2)
	

	
	
	'''
	inputs2 = tf.keras.Input(shape=(6, 32, 32, 1))
	x = layers.ConvLSTM2D(
		filters=32,
		kernel_size=(3,3),
		dilation_rate=(3,3),
		activation="relu"
	)(inputs2)
	x = Flatten()(x)
	x = RepeatVector(1)(x)
	x = LSTM(64, activation='relu', return_sequences=True)(x)
	lstm2 = TimeDistributed(Dense(100, activation='relu'))(x)
	#lstm2 = TimeDistributed(Dense(1))(x)
	lstm2 = Flatten()(lstm2)
	'''
	
	
	'''
	inputs2 = tf.keras.Input(shape=(192,32,1))
	conv = Conv2D(32, kernel_size=(2,2),dilation_rate=(3,3),  activation='relu', input_shape=(192,32,1), data_format="channels_last")(inputs2)
	conv = MaxPooling2D(pool_size=(2, 2))(conv)
	conv = Conv2D(32, kernel_size=(2,2),dilation_rate=(3,3), activation='relu')(conv)
	conv = MaxPooling2D(pool_size=(2, 2))(conv)
	conv = Conv2D(32, 1, dilation_rate=(3,3), activation='relu')(conv)
	conv = MaxPooling2D(pool_size=(2, 2))(conv)
	conv = Conv2D(32, 1, activation='relu')(conv)
	lstm2 = Flatten()(conv)
	'''

	
	#merge = concatenate([lstm1, lstm2])
	merge = lstm1
	#merge = Dense(16, activation='relu')(merge)
	merge = Dense(64, activation='relu')(merge)
	#merge = Dense(64, activation='relu')(merge)
	
	#merge = Dense(64, activation='relu')(merge)
	merge = Dense(1)(merge)
	
	
	model = Model(inputs=[inputs1, inputs2], outputs=merge)
	print(model.summary())
	
	#sys.exit()
	

	
	opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
	model.compile( loss='mae', optimizer=opt, metrics=['mse','mae'])
	
	#model.fit_generator(data_generators(), steps_per_epoch = 50, epochs=300, shuffle=True)
	#model.fit(data_generators(), batch_size = 500, epochs=10, shuffle=True)
	
	# Instantiate an optimizer to train the model.
	optimizer = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
	#loss_fn = tfa.losses.PinballLoss(tau=.6)
	loss_fn = tf.keras.losses.MeanAbsoluteError()
	#loss_fn = tf.keras.losses.MeanSquaredError()

	
	for runs in range(20):
		count = 0
		x, y, I1 = [], [], []
		for i in data_generators():
			#print(count)
			x1 = np.asarray(i[0][1])
			#print(x1.shape)
			#x1 = np.reshape(x1, (6,32, 32,1))
			
			x2 = i[0][0]
			x2 = np.reshape(x2, (steps,15))
			
			I1.append(x1)
			x.append(x2)
			y.append(i[1])
			
			
			count += 1
			if count == 300:
				break
		x = np.asarray(x, dtype='float')
		y = np.asarray(y, dtype='float')		
		I1 = np.asarray(I1, dtype='float')
		
		
		'''
		with tf.GradientTape() as tape:
			logits = model([x, I1], training=True)
			loss_value = loss_fn(y, logits)
		grads = tape.gradient(loss_value, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))
		print("Training loss (for one batch) at step %d: %.4f"  % (runs, float(loss_value))  )
		'''
		
		#model.train_on_batch([x, I1],y)
		print(model.train_on_batch([x, I1], y))
		
	
	MAD = []
	MAD1 = []
	mae = []
	PRED = []
	
	count = 0
	
	
	for i in os.listdir(dire+'/test'):
		x, y, imaged, check = PrepareTestData(N, steps, i)
		if check:
			x = np.reshape(x, (1,steps,15))
			preds = model.predict([x, imaged])
			preds = preds.flatten()
			preds = np.reshape(preds, (1,-1))
			preds = scalerY.inverse_transform(preds)
			preds = preds.flatten()
			act = np.reshape(y, (1,-1))
			act = scalerY.inverse_transform(act)
			act = act.flatten()
			mad = abs(preds-act)
			
			mad1 = stats.median_abs_deviation([act,preds], scale='normal')
			#print(mad1)
			mad1 = np.mean(mad1)
			mad = np.mean(mad)
			
			#print (mad)
			mae.append(mad)
			MAD.append(max(0,100-mad))
			MAD1.append(max(0, 100-mad1))
			PRED.append([act, preds])


	MAD1 = np.asarray(MAD1)
	MAD = np.asarray(MAD)
	mae = np.asarray(mae)
	print(np.mean(MAD))
	print(np.mean(MAD1))
	print(np.mean(mae))
	
	print('\n\n')
	
	MAD = []
	MAD1 = []
	mae = []
	PRED = []
	
	count = 0
	
	
	for i in os.listdir(dire+'/test'):
		x, imaged = PrepareTestDataMain(N, steps, i)
		#print(x.shape)
		x = np.reshape(x, (1,steps,15))
		preds = model.predict([x, imaged])
		preds = preds.flatten()
		preds = np.reshape(preds, (1,-1))
		preds = scalerY.inverse_transform(preds)
		preds = preds.flatten()
		#print(preds)
		PRED.append([i, preds[0]])
		#sys.exit()
		#time.sleep(11)
	
	PRED = sorted(PRED, key = lambda x:int(x[0]))
	
	out = open(str(120)+'-'+str(steps),'w')
	for i in PRED:
		out.write(str(i[0])+' '+str(i[1])+'\n')
	out.close()	
	
	
	sys.exit()
	
	
	
	
	
	
	
	
	
	
	
