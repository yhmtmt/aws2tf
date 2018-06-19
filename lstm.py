import tensorflow as tf
import random
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import pdb
import csv
import ld
pdb.set_trace()

# 8 inputs [eng, rud, rroll, rpitch, ryaw, rcog, sog, rsog]
# 6 outputs [rroll, rpitch, ryaw, rcog, sog, rsog][10]

num_of_input_nodes = 8
num_of_hidden_nodes = 128
num_of_lstm_layers = 2
num_of_prediction_steps = 5
num_of_output_nodes = 6 * num_of_prediction_steps
length_of_sequences = 64
num_of_training_epochs = 20000
batch_size = 128
learning_rate = 0.001
forget_bias = 0.8
num_of_sample = 1000
do_train = True
refine_existing_training = True
model_path = "/home/ubuntu/matumoto/model/"
model_name = "lstm_%dx%d_i%d_o%d.ckpd" % (num_of_hidden_nodes, num_of_lstm_layers, num_of_input_nodes, num_of_output_nodes)

print("Model:"+ model_path + model_name)

def inference(input_ph):
    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal(
            [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        in1 = tf.transpose(input_ph, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
        #in2 = tf.reshape(input_ph, [-1, num_of_input_nodes])
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        #in4 = tf.split(in3, batch_size, 0)
        in4 = tf.reshape(in3, [length_of_sequences, -1, num_of_hidden_nodes])
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=True)

#        cell = [tf.nn.rnn_cell.BasicLSTMCell(n, forget_bias=forget_bias, state_is_tuple=True) for n in num_of_hidden_nodes]
#        cell = [tf.nn.rnn_cell.LSTMCell(n, forget_bias=forget_bias, state_is_tuple=True) for n in num_of_hidden_nodes]
        if(do_train):
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.8)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_of_lstm_layers, state_is_tuple=True)
        #init_state = cell.zero_state(batch_size, tf.float32)
        rnn_output, states_op = tf.nn.dynamic_rnn(cell, inputs=in4, dtype=tf.float32, time_major=True)
        
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        # Add summary ops to collect data
        w1_hist = tf.summary.histogram("weights1", weight1_var)
        w2_hist = tf.summary.histogram("weights2", weight2_var)
        b1_hist = tf.summary.histogram("biases1", bias1_var)
        b2_hist = tf.summary.histogram("biases2", bias2_var)
        output_hist = tf.summary.histogram("output",  output_op)
        results = [weight1_var, weight2_var, bias1_var,  bias2_var]
        return output_op, states_op, results

def calc_accuracy(output_op, inputs, ts, prints=False):
    pred_dict = {
        input_ph:  inputs,
        supervisor_ph: ts,
     }
    output = sess.run([output_op], feed_dict=pred_dict)

    loss = np.mean(abs(output - ts))

    print("loss %f" % loss)
    return output

args = sys.argv

if len(args) == 1:
    sys.exit()

fcsv_names, fsizes = ld.loadAWS1DataList(args[1])

print("Files")
print(fcsv_names)
print(fsizes)

if not do_train:
    length_of_sequences = 1
    batch_size = 1

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

with tf.Graph().as_default():
    input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
    output_op, states_op, datas_op = inference(input_ph)
    if do_train:
        loss_op = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        training_op = optimizer.minimize(loss_op)
        tf.summary.scalar("loss", loss_op)

    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if(refine_existing_training):
            if(tf.train.checkpoint_exists(model_path)):
                saver.restore(sess, model_path + model_name)
            else:
                print("check point " + model_path + model_name + " not there")

        summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
        sess.run(init)

        if do_train:
            for epoch in range(num_of_training_epochs):
                inputs, supervisors = ld.getAWS1DataBatch(fcsv_names, fsizes, batch_size, length_of_sequences, num_of_prediction_steps)
                train_dict = {
                    input_ph:      inputs,
                    supervisor_ph: supervisors,
                }
                train_loss, train_state, _ = sess.run([loss_op, states_op, training_op], feed_dict=train_dict)

                if (epoch) % 100 == 0:
                    summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                    print("train#%d, train loss: %e" % (epoch, train_loss))
                    summary_writer.add_summary(summary_str, epoch)
                    if (epoch) % 500 == 0:
                        xse, tse =  ld.getAWS1DataBatch(fcsv_names, fsizes, batch_size, length_of_sequences, num_of_prediction_steps, True)
                        calc_accuracy(output_op, xse, tse)

            xse, tse = ld.getAWS1DataBatch(fcsv_names, fsizes, batch_size, length_of_sequences, num_of_prediction_steps, True)
            calc_accuracy(output_op, xse, tse, prints=True)
            datas = sess.run(datas_op)
            saver.save(sess, model_path + model_name)
        else:
            ifile=0
            isection=1
            ipos=0
            loss = np.array([0] * num_of_output_nodes * num_of_prediction_steps)
            loss_total= 0
            count=0
            t,eng,rud,roll,rroll,pitch,rpitch,yaw,ryaw,cog,rcog,sog,rsog = ld.loadAWS1Data(fcsv_names[ifile]) 
            while(ifile < len(fcsv_names)):
                xse, tse = ld.getAWS1BatchSectionSeq(num_of_prediction_steps, eng, rud, rroll, rpitch, ryaw, rcog, sog, rsog, isection * 0.25, min(1.0, (isection + 1) * 0.25))
                if(xse == None and tse == None):
                    if(isection == 1):
                        isection = 3
                    else:
                        ifile += 1
                        t,eng,rud,roll,rroll,pitch,rpitch,yaw,ryaw,cog,rcog,sog,rsog = ld.loadAWS1Data(fcsv_names[ifile]) 
                        isection = 1
                    continue

                pred_dict = {
                    input_ph:  xse,
#                    supervisor_ph: tse,
                }
                output = sess.run([output_op], feed_dict=pred_dict)
                abserr = abs(output - tse)
                loss += abserr
                loss_total += np.mean(abserr)
                count += 1
                ipos += 1

            loss /= float(count)
            loss = np.reshape(loss, [num_of_prediction_steps, num_of_output_nodes])
            loss_total /= float(count)

            print("loss_total %f" % loss_total)
            print("t, rroll, rpitch, ryaw, rcog, sog, rsog")
            for i in range(0, num_of_prediction_steps):
                print("%d,%f,%f,%f,%f,%f,%f" % (i+1, loss[i][0], loss[i][1], loss[i][2], loss[i][3], loss[i][4], loss[i][5]))

sys.exit()
