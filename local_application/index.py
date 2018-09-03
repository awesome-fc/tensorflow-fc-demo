# -*- coding:utf-8 -*-   
import os
import sys
import cv2  
import numpy as np
import tensorflow as tf  

saver = None

def reversePic(src):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src[i, j] = 255 - src[i, j]
    return src 

def main():
    sess = tf.Session()  

    saver = tf.train.import_meta_graph('model_data/model.meta')
 
    saver.restore(sess, 'model_data/model')
    graph = tf.get_default_graph()
    
    input_x = sess.graph.get_tensor_by_name("Mul:0")
    y_conv2 = sess.graph.get_tensor_by_name("final_result:0")
    
    path="pic/e2.jpg"  
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    im = reversePic(im)

    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)  

    x_img = np.reshape(im , [-1 , 784])  
    output = sess.run(y_conv2 , feed_dict={input_x:x_img})  
    print 'the predict is %d' % (np.argmax(output)) 

    sess.close()

if __name__ == '__main__':  
    main()  
