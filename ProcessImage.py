from defisheye import Defisheye
import time
import numpy as np
import sys
import skimage
from skimage import io
from skimage.color import rgb2gray
from skimage import data
from skimage import filters
import os


# https://www.frontiersin.org/files/Articles/722212/fenrg-09-722212-HTML-r1/image_m/fenrg-09-722212-g004.jpg

'''
A Deep Learning Model to Forecast Solar Irradiance Using a
Sky Camera	
'''

dire = os.getcwd()

for i in os.listdir(dire+'/test'):
	#print(i)
	if os.path.isdir(dire+'/test/'+i):
		#print('aa')
		for j in os.listdir(dire+'/test/'+i):
			if '.jpg' in j:
				if 'san' not in j:
					
					print (dire+'/test/'+i)

					dtype = 'linear'
					format = 'fullframe'
					fov = 180
					pfov = 120

					img = dire+'/test/'+i+'/'+j
					img_out = dire+'/test/'+i+'/'+j[:-4]+'_sant.jpg'

					obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
					obj.convert(img_out)
					
					sky = io.imread(img_out)
					red = np.double(sky[:,:,0])
					blue = np.double(sky[:,:,2])

					num = blue-red
					den = blue+red
					ratio = num/den
					sky_grey = ratio*255
					
				
					
					
					sky_change1 = sky_grey < 50
					sky_change1 = sky_change1*255

					sky_change2 = sky_grey < 50
					sky_change2 = sky_change2*255

					sky_change3 = sky_grey < 50
					sky_change3 = sky_change3*255

					arr = []
					for k in range(len(sky_change1)):
						arr1 = []
						arr1.append(sky_change1[k])
						arr1.append(sky_change2[k])
						arr1.append(sky_change3[k])
						arr1 = np.asarray(arr1, dtype='int')
						#print (arr1)
						arr1 = np.mean(arr1, axis=0)
						arr.append(arr1)
					
					sky_change1 = np.asarray(arr)
					
					
					
					io.imsave(img_out, sky_change1)
					


'''
we enable to observe that the RGB values of blue sky, white cloud, and dark gray
46 cloud are about (0, 0, 255), (255, 255, 255), (180, 180, 180) respectively. 

'''


