import numpy as np
import tensorflow as tf
import math
#from matplotlib import pyplot as plt
import subprocess

dims=10
no_epochs = 1000
batch_size=100
x_range=10
dataset_size=1000

l=0.07

def linear_regression():

	x = tf.placeholder(tf.float32,shape=(None,dims))
	y = tf.placeholder(tf.float32,shape=(None,1))

	#w=tf.Variable(tf.random_normal([dims,1]))
	#w=tf.Variable(np.array([[1,-1/math.factorial(3),1/math.factorial(5),-1/math.factorial(7),1/math.factorial(9),-1/math.factorial(11),1/math.factorial(13),-1/math.factorial(15),1/math.factorial(17),-1/math.factorial(19)]]).T.astype('float32'))
	#w=tf.Variable(np.array([[1,-1/math.factorial(3),1/math.factorial(5),-1/math.factorial(7),1/math.factorial(9)]]).T.astype('float32'))
	w=tf.Variable(tf.zeros([dims,1]))
	#b=tf.Variable(tf.random_normal([1]))
		
	pred=tf.matmul(x,w)
	#pred=tf.add(tf.matmul(x,w),b)
	
	cost = tf.reduce_mean(tf.square(y-pred)) + l*tf.nn.l2_loss(w)
	#cost = tf.reduce_mean(tf.pow(pred-Y,2))	

	return x,y,w,pred,cost

'''
def plot(w,no):
	#x=np.ndarray((dims,))
	x_t=np.arange(-10,10,0.1).T
	x=np.ndarray((x_t.shape[0],dims))
	x[:,0]=x_t.T
	temp=x[:,0]
	for i in range(1,dims):
		x[:,i]=x[:,i-1]*temp

	y=np.dot(x,w)
	
	plt.figure(1)
	plt.plot(x_t,y,marker='.',markersize=0.01)
	plt.savefig("img/{}.png".format(no))
	plt.clf()
'''

def run():

	subprocess.run(["bash","-c","\"rm img/*\""])

	x,y,w,pred,cost=linear_regression()

	#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
	optimizer = tf.train.AdagradOptimizer(0.01).minimize(cost)
	#optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		x_t,y_d = generate_data(dataset_size,x_range)
		x_d=np.zeros((x_t.shape[0],dims))
		x_d[:,0]=x_t.T
		temp=x_d[:,0]
		for i in range(1,dims):
			x_d[:,i]=x_d[:,i-1]*temp	


		check_op = tf.add_check_numerics_ops()
		
		for epoch in range(no_epochs):
			epoch_loss=0
			a=[i for i in range(dataset_size)]
			np.random.shuffle(a)
			x_d=x_d[a,:]
			y_d=y_d[a,:]

			for batch in range(int(dataset_size/batch_size)):

				x_batch=x_d[batch*batch_size:(batch+1)*batch_size]
				y_batch=y_d[batch*batch_size:(batch+1)*batch_size]
				try:
					_,loss,_,value_w=sess.run([optimizer,cost,check_op,w], feed_dict = {x:x_batch,y:y_batch})
				except tf.errors.InvalidArgumentError as e:
					print(e.op)
					exit()
				epoch_loss+=loss
				#print(epoch_loss)
				#print(value_w)
				#plot(value_w)

			print('Epoch', epoch, 'loss:',epoch_loss)
			#print(value_w)

		plot(value_w,epoch)
		#print('Error:',error.eval({x:test_x_data,y:test_y_data}))

def generate_data(n,range):
	#Function y = sin(x)
	x=np.random.rand(n,1)
	x*=2*range
	x-=range
	y=np.sin(x)
	noise=np.random.randn(n,1) * 0.01
	y+=noise
	#plt.figure(1)
	#plt.scatter(x,y,marker='.')
	#plt.show()
	return x,y


def main():
	run()


if __name__ == '__main__':
	main()
gputest_2.py
Open with Drive Notepad
Displaying gputest_2.py.