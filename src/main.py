import tensorflow as tf
import numpy as np
import pandas as pd
import math

n_features=10
n_hidden=20
batch_size = 64
epochs = 1000
tr_test_fraction = 0.7
validation_tr_fraction = 0.2
learning_rate = 0.003

def nn_model():

    x = tf.placeholder(tf.float32,shape=(None,n_features)) #Specifies shape of the tensor
    y = tf.placeholder(tf.float32,shape=(None,1))

    w1=tf.Variable(tf.random_uniform([n_features,n_hidden]))
    b1=tf.Variable(tf.random_uniform([1,n_hidden]))

    w2=tf.Variable(tf.random_uniform([n_hidden,1]))
    b2=tf.Variable(tf.random_uniform([1,1]))
    
    pred = tf.add(tf.matmul(x,w1),b1)
    pred = tf.sigmoid(pred)

    pred = tf.add(tf.matmul(pred,w2),b2)
    pred = tf.sigmoid(pred)

    loss =  tf.losses.sigmoid_cross_entropy(y,pred)
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y)
    accuracy = tf.metrics.accuracy(y,tf.round(pred))   
    return x,y,pred,loss,accuracy

def read_data(file):
    df = pd.read_csv(file)
    df.drop(labels=['RowNumber', 'UID', 'Customer_name'], axis=1, inplace=True)
    city_map = {'Hyderabad': 1, 'Pilani': 2, 'Goa': 3}
    gender_map = {'Male': -1, 'Female': 1}
    
    nrows = len(df.index)
    city_idx = df.columns.get_loc('City')
    gender_idx = df.columns.get_loc('Gender')
    for i in range(nrows):
        df.iloc[i, city_idx] = city_map[df.iloc[i, city_idx]]
        df.iloc[i, gender_idx] = gender_map[df.iloc[i, gender_idx]]
    return df
    
def main():
    df = read_data("../data/raw_data.csv")
    print("data read from csv...")
    
    """
    To partition the entire dataset as training and testing data
    """
    data_size = len(df.values)
    train_data_size = math.floor(data_size*tr_test_fraction)
    data = df.values
    #np.random.shuffle(data)
    train_data = data[:train_data_size]
    test_data = data[train_data_size:]
    train_data_size = len(train_data)
  
    x,y,pred,loss,acc = nn_model()

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for ep in range(epochs):
            epoch_loss=0

            np.random.shuffle(train_data)
            vdata_size = math.floor(validation_tr_fraction*train_data_size)
            vdata = train_data[:vdata_size]
            tdata = train_data[vdata_size:]
            tdata_size = len(tdata)


            no_of_batches = math.floor(tdata_size/batch_size)
            
            batch_x = np.zeros(shape=(batch_size,n_features), dtype = np.float32)
            batch_y = np.zeros(shape=(batch_size,1), dtype = np.float32)
            
            for i in range(no_of_batches):
                batch_x=tdata[i*batch_size:(i+1)*batch_size,:-1]
                batch_y=tdata[i*batch_size:(i+1)*batch_size,-1:]
                eloss,_ = sess.run([loss,optimizer],feed_dict={x:batch_x,y:batch_y})
                epoch_loss+=eloss
            
            vloss,vacc=sess.run([loss,acc],feed_dict={x:vdata[:,:-1],y:vdata[:,-1:]})
            
            #print("Epoch Loss:{}".format(epoch_loss),end=' ')
            print("Epoch:{}\tValidation Loss:{}\tValidation Accurracy:{}".format(ep,vloss,vacc[0]))
        
        tloss,tacc=sess.run([loss,acc],feed_dict={x:test_data[:,:-1],y:test_data[:,-1:]})
        print("Test Loss:{}\tTest Accurracy:{}".format(tloss,tacc[0]))

if __name__=="__main__":
    main()