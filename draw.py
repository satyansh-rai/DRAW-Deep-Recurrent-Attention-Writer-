

""""
DRAW Implementation on MNIST Dataset
Satyansh Rai 2016B4A70632P
Aaryan Kapoor 2016B4A70166P
Parth Misra 2016B5A70560P
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os

tf.flags.DEFINE_string("data_dir", "", "")
FLAGS = tf.flags.FLAGS

enc = 256 
dec = 256
size_z=10 
T=10 
batch_size=64 #Mini batch Gradient Descent
train_iterations=20000
eta=1e-3
epsilon=1e-8 
A,B = 28,28 
image_size = B*A 
read_glimpse = 2
write_glimpse = 5 
read_size = 2*read_glimpse*read_glimpse 
write_size = write_glimpse*write_glimpse 

lstm_encoder = tf.contrib.rnn.LSTMCell(enc, state_is_tuple=True) #creating RNN of size 256 for encoder
lstm_decoder = tf.contrib.rnn.LSTMCell(dec, state_is_tuple=True) #creating RNN of size 256 for decoder
x = tf.placeholder(tf.float32,shape=(batch_size,image_size))
e=tf.random_normal((batch_size,size_z), mean=0, stddev=1) #noise

SHARE=None

def linear(inp,output_dim):
    
    weight=tf.get_variable("w", [inp.get_shape()[1], output_dim]) 
    bias=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(inp,weight)+bias

def filterb(gx, gy, sigma2,delta, N):
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    av_x = gx + (grid_i - N / 2 - 0.5) * delta 
    av_y = gy + (grid_i - N / 2 - 0.5) * delta 
    av_x = tf.reshape(av_x, [-1, N, 1])
    av_y = tf.reshape(av_y, [-1, N, 1])
    
    Fx = tf.exp(-tf.square(a - av_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - av_y) / (2*sigma2)) 
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),epsilon)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),epsilon)
    return Fx,Fy

def encode(state,input):
   
    with tf.variable_scope("encoder",reuse=SHARE):
        return lstm_encoder(input,state)

def decode(state,input):
    with tf.variable_scope("decoder",reuse=SHARE):
        return lstm_decoder(input, state)


def attn(scope,h_decoder,N):
    with tf.variable_scope(scope,reuse=SHARE):
        parameters=linear(h_decoder,5)
    
    gx_,gy_,log_sigma,log_delta,log_gamma=tf.split(parameters,5,1)
    grid_x=(A+1)/2*(gx_+1)
    grid_y=(B+1)/2*(gy_+1)
    sigma=tf.exp(log_sigma)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) 
    return filterb(grid_x,grid_y,sigma,delta,N)+(tf.exp(log_gamma),)

def read(x,x_err,h_decoder_prev):
    Fx,Fy,gamma=attn("read",h_decoder_prev,read_glimpse)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_glimpse) 
    x_err=filter_img(x_err,Fx,Fy,gamma,read_glimpse)
    return tf.concat([x,x_err], 1) 

def sampleQ(h_encoder):
    
    with tf.variable_scope("mu",reuse=SHARE):
        mu=linear(h_encoder,size_z)
    with tf.variable_scope("sigma",reuse=SHARE):
        logsigma=linear(h_encoder,size_z)
        sigma=tf.exp(logsigma)
    return (mu + e*sigma, mu, logsigma, sigma)

 

def write(h_decoder):
    with tf.variable_scope("writeW",reuse=SHARE):
        w=linear(h_decoder,write_size)
    N=write_glimpse
    w=tf.reshape(w,[batch_size,N,N])
    Fx,Fy,gamma=attn("write",h_decoder,write_glimpse)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.matmul(Fyt,tf.matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,B*A])
   
    return wr*tf.reshape(1.0/gamma,[-1,1])

cs=[0]*T
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T 

h_decoder_prev=tf.zeros((batch_size,dec))
encoder_state=lstm_encoder.zero_state(batch_size, tf.float32)
decoder_state=lstm_decoder.zero_state(batch_size, tf.float32)


for t in range(T):
    c_prev = tf.zeros((batch_size,image_size)) if t==0 else cs[t-1]
    x_err=x-tf.sigmoid(c_prev) 
    r=read(x,x_err,h_decoder_prev)
    h_encoder,encoder_state=encode(encoder_state,tf.concat([r,h_decoder_prev], 1))
    z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_encoder)
    h_decoder,decoder_state=decode(decoder_state,z)
    cs[t]=c_prev+write(h_decoder)
    h_decoder_prev=h_decoder
    SHARE=True

 

def crossentropy(t,o):
    return -(t*tf.log(o+epsilon) + (1.0-t)*tf.log(1.0-o+epsilon))


x_recons=tf.nn.sigmoid(cs[-1])


Lx=tf.reduce_sum(crossentropy(x,x_recons),1) 
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-.5 
KL=tf.add_n(kl_terms) 
Lz=tf.reduce_mean(KL) 

cost=Lx+Lz



optimizer=tf.train.AdamOptimizer(eta, beta1=0.5)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v)
train_op=optimizer.apply_gradients(grads)

""" To run the training dataset"""

directory = os.path.join(FLAGS.data_dir, "mnist")
if not os.path.exists(directory):
	os.makedirs(directory)
train_data = mnist.input_data.read_data_sets(directory, one_hot=True).train

fetches=[]
fetches.extend([Lx,Lz,train_op])
Lx1=[0]*train_iterations
Lz1=[0]*train_iterations

session=tf.InteractiveSession()

saver = tf.train.Saver()
tf.global_variables_initializer().run()


for i in range(train_iterations):
	xtrain,_=train_data.next_batch(batch_size) 
	feed_dict={x:xtrain}
	results=session.run(fetches,feed_dict)
	Lx1[i],Lz1[i],_=results
	if i%100==0:
		print("iter=%d : Lx: %f Lz: %f" % (i,Lx1[i],Lz1[i]))



canvases=session.run(cs,feed_dict) 
canvases=np.array(canvases) 

out_file=os.path.join(FLAGS.data_dir,"draw_data1.npy")
np.save(out_file,[canvases,Lx1,Lz1])
print("Outputs saved in file: %s" % out_file)

ckpt_file=os.path.join(FLAGS.data_dir,"drawmodel1.ckpt")
print("Model saved in file: %s" % saver.save(session,ckpt_file))

session.close()
