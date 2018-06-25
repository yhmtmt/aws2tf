import tensorflow as tf
import random
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import pdb
import csv
import ld
pdb.set_trace()

# 13 inputs [eng, rud, rev, roll, rroll, pitch, rpitch, yaw, ryaw, cog, rcog, sog, rsog]
# 6 outputs [rroll, rpitch, ryaw, rcog, sog, rsog]
str_vec_in = ["eng", "rud", "rev", "roll", "rroll", "pitch", "rpitch", "yaw", "ryaw", "cog", "rcog", "sog", "rsog"]
str_vec_out = ["rroll", "rpitch", "ryaw", "rcog", "sog", "rsog"]
str_vec = list(set(str_vec_in + str_vec_out))
str_vec_in
str_vec_out
str_vec

dim_vec_in = len(str_vec_in)
dim_vec_out = len(str_vec_out)
num_nodes_in = dim_vec_in
num_hidden_nodes = 128
num_lstm_layers = 2
len_pred = 32
num_nodes_out = dim_vec_out * len_pred
length_of_sequences = 128
num_training_epochs = 200000
batch_size = 128
learning_rate = 0.001
forget_bias = 0.8

do_train = False
refine_existing_training = True
model_path = "/home/ubuntu/matumoto/model/"
plot_path="/home/ubuntu/matumoto/plot/"
model_name = "lstm_%dx%d_i%d_o%d" % (num_hidden_nodes, num_lstm_layers, num_nodes_in, num_nodes_out)

print("Model:"+ model_path + model_name + ".ckpd")

def inference(input_ph):
    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal(
            [num_nodes_in, num_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal(
            [num_hidden_nodes, num_nodes_out], stddev=0.1), name="weight2")
        bias1_var = tf.Variable(tf.truncated_normal([num_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([num_nodes_out], stddev=0.1), name="bias2")

        in1 = tf.transpose(input_ph, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, num_nodes_in])
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.reshape(in3, [length_of_sequences, -1, num_hidden_nodes])
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_nodes, forget_bias=forget_bias, state_is_tuple=True)

        if(do_train):
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.8)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_lstm_layers, state_is_tuple=True)
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
    input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, num_nodes_in], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_nodes_out], name="supervisor")
    output_op, states_op, datas_op = inference(input_ph)
    if do_train:
        loss_op = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        training_op = optimizer.minimize(loss_op)
        tf.summary.scalar("loss", loss_op)

    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        if(refine_existing_training):
            if(tf.train.checkpoint_exists(model_path+model_name+".ckpd")):
                saver.restore(sess, model_path + model_name+".ckpd")
            else:
                print("check point " + model_path + model_name + ".ckpd not there")

        summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)

        if do_train:
            for epoch in range(num_training_epochs):
                inputs, supervisors = ld.getAWS1DataBatch(fcsv_names, str_vec_in, str_vec_out, fsizes, batch_size, length_of_sequences, len_pred)
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
                        xse, tse =  ld.getAWS1DataBatch(fcsv_names, str_vec_in, str_vec_out, fsizes, batch_size, length_of_sequences, len_pred, True)
                        calc_accuracy(output_op, xse, tse)

            xse, tse = ld.getAWS1DataBatch(fcsv_names, str_vec_in, str_vec_out, fsizes, batch_size, length_of_sequences, len_pred, True)
            calc_accuracy(output_op, xse, tse, prints=True)
            datas = sess.run(datas_op)
            saver.save(sess, model_path + model_name + ".ckpd")
        else:
            ifile=0
            isection=1
            ipos=0
            loss = np.array([0.] * num_nodes_out)
            loss_total= 0.
            count=0
            print("Loading %s" % fcsv_names[ifile])
            aws1_data = ld.loadAWS1Data(fcsv_names[ifile],str_vec) 

            sdata = [[],[],[],[],[],[]]
            pdata = [[],[],[],[],[],[]]

            while(ifile < len(fcsv_names)):
                xse, tse = ld.getAWS1BatchSectionSeq(str_vec_in, str_vec_out, len_pred, aws1_data, ipos, isection * 0.25, min(1.0, (isection + 1) * 0.25))
                if(xse is None and tse is None):
                    time_axis = [i for i in range(len(sdata[0]))]
                    for idata in range(len(sdata)):
                        plt.subplot(2,1,1)
                        plt.plot(np.array(time_axis), np.array(pdata[idata]))
                        plt.subplot(2,1,2)
                        plt.plot(np.array(time_axis), np.array(sdata[idata]))
                        figname=model_name+"-%d-%d-%s" % (ifile, isection, str_vec_out[idata])
                        plt.savefig(plot_path + figname + ".png")
                        plt.clf()

                    if(isection == 1):
                        isection = 3
                    else:
                        ifile += 1
                        print("%d samples" % count)                        
                        if(len(fcsv_names[ifile])==0):
                            break
                        print("Loading %s" % fcsv_names[ifile])
                        aws1_data = ld.loadAWS1Data(fcsv_names[ifile], str_vec) 
                        isection = 1
                    ipos = 0
                    continue

                pred_dict = {
                    input_ph:  xse,
                }
                output = sess.run([output_op], feed_dict=pred_dict)
                abserr = abs(output - tse)
                for idata in range(dim_vec_out):
                    pdata[idata].append(output[0][0][idata])
                    sdata[idata].append(tse[0][idata])

                loss += abserr[0][0]
                loss_total += np.mean(abserr)
                count += 1
                ipos += 1

            loss /= float(count)
            loss = np.reshape(loss, [len_pred, dim_vec_out])
            loss_total /= float(count)

            print("loss_total %f" % loss_total)
            print("t, rroll, rpitch, ryaw, rcog, sog, rsog")
            for i in range(0, len_pred):
                print("%d,%f,%f,%f,%f,%f,%f" % (i+1, loss[i][0], loss[i][1], loss[i][2], loss[i][3], loss[i][4], loss[i][5]))

sys.exit()
