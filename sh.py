import numpy as np;
import gzip

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

data = data.reshape(-1,1,28,28)

data = data/np.float32(256)

print(data[0][0])


import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
plt.gray()
for i in range(6):
	plt.show(plt.imshow(data[i][0]))
'''

for i in range(60000):
	plt.imshow(data[i][0])
	plt.savefig('./images/MNIST_IMAGE'+str(i)+'.png')#save MNIST image
#	plt.show()#Show / plot that image


 '''
