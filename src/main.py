import tensorflow as tf
import numpy as np
import pandas as pd

n_features=9

n_hidden=10

def nn_model():

    x = tf.placeholder(tf.float32,shape=(None,n_features))
    y = tf.placeholder(tf.float32,shape=(None,1))

    w1=tf.Variable(tf.random_normal([n_features,n_hidden]))
    b1=tf.Variable(tf.random_normal([1,n_hidden]))

    w2=tf.Variable(tf.random_normal([n_hidden,1]))
    b2=tf.Variable(tf.random_normal([1,1]))
    
    pred = tf.add(tf.matmul(x,w1),b1)
    pred = tf.sigmoid(pred)

    pred = tf.add(tf.matmul(pred,w2),b2)
    pred = tf.sigmoid(pred)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(pred,y)

    return x,y,pred,loss

def read_data(file):
    df = pd.read_csv(file)
    df.drop(labels=['RowNumber', 'UID', 'Customer_name'], axis=1, inplace=True)
    city_map = {'Hyderabad': 0, 'Pilani': 1, 'Goa': 2}
    gender_map = {'Male': 0, 'Female': 1}
    
    nrows = len(df.index)
    city_idx = df.columns.get_loc('City')
    gender_idx = df.columns.get_loc('Gender')
    for i in range(nrows):
        df.iloc[i, city_idx] = city_map[df.iloc[i, city_idx]]
        df.iloc[i, gender_idx] = gender_map[df.iloc[i, gender_idx]]

    return df
    
def main():

    df = read_data("../data/raw_data.csv")
    train_data = df.values
    print(type(train_data))

    x,y,pred,loss = nn_model()

    optimizer = tf.train.AdamOptimizer(0.5).minimize(loss)

    with sess as tf.InteractiveSession:
        tf.global_variables_initializer().run()
        for ep in range(epochs):
            for i in range(data_size/batch_size):
                # fil batch_x batch_y
                loss_val,_ =sess.run([loss,optimizer],feed_dict={x:batch_x,y:batch_y})
        
            #y_pred_batch = session.run(y_pred, {x: x_batch})
        

        
if __name__=="__main__":
    main()