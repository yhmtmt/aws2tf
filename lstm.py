import tensorflow as tf
import random
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import pdb
import csv
import ld
import argparse
pdb.set_trace()

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
        in4 = tf.reshape(in3, [len_seq, -1, num_hidden_nodes])
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

# 13 inputs [eng, rud, rev, roll, rroll, pitch, rpitch, yaw, ryaw, cog, rcog, sog, rsog]
# 6 outputs [rroll, rpitch, ryaw, rcog, sog, rsog]
str_vec_in = ["eng", "rud", "rev", "roll", "rroll", "pitch", "rpitch", "yaw", "ryaw", "cog", "rcog", "sog", "rsog"]
str_vec_out = ["rroll", "rpitch", "ryaw", "rcog", "sog", "rsog"]
fac_vec_in=[1.0/255.0, 1.0/255.0, 1.0/5500, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/20.0, 1.0/180.0, 1.0/180.0, 1.0/360.0, 1.0/180.0, 1.0/15.0, 1.0/5.0]
fac_vec_out=[1.0/20.0, 1.0/20.0, 1.0/180.0, 1.0/180.0, 1.0/15.0, 1.0/5.0]
#str_vec_out = ["sog", "rsog"]
#fac_vec_out=[1.0/15.0, 1.0/5.0]

str_vec = list(set(str_vec_in + str_vec_out))

num_hidden_nodes = 64
num_lstm_layers = 2
len_pred = 32
len_seq = 256
num_training_epochs = 50000
batch_size = 128
learning_rate = 0.001
forget_bias = 0.8
do_train = False
refine_existing_model = True
model_path = "/home/ubuntu/matumoto/model/"
plot_path="/home/ubuntu/matumoto/plot/"


parser = argparse.ArgumentParser()
if(sys.stdin.isatty()):
    parser.add_argument("list_file", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--hidden", type=int)
parser.add_argument("--layer", type=int)
parser.add_argument("--batch", type=int)
parser.add_argument("--epoch", type=int)
parser.add_argument("--seq", type=int)
parser.add_argument("--pred", type=int)
args = parser.parse_args()

if not hasattr(args, "list_file"):    
    sys.exit()
else:
    flist = args.list_file
        
if args.train:
    do_train = True

if args.retrain:
    refine_existing_model = True

if args.hidden:
    num_hidden_nodes = args.hidden

if args.layer:
    num_lstm_layers = args.layer

if args.batch:
    batch_size = args.batch

if args.epoch:
    num_training_epochs = args.epoch

if args.seq:
    len_seq = args.seq


dim_vec_in = len(str_vec_in)
dim_vec_out = len(str_vec_out)
num_nodes_in = dim_vec_in
num_nodes_out = dim_vec_out * len_pred
model_name = "lstm_%dx%d_i%d_o%d_seq%d_epoch%d" % (num_hidden_nodes, num_lstm_layers, num_nodes_in, num_nodes_out, len_seq, num_training_epochs)

print("Model:"+ model_path + model_name + ".ckpd")

fcsv_names, fsizes = ld.loadAWS1DataList(flist)

if not do_train:
    len_seq = 1
    batch_size = 1

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

with tf.Graph().as_default():
    input_ph = tf.placeholder(tf.float32, [None, len_seq, num_nodes_in], name="input")
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
        if(refine_existing_model):
            if(tf.train.checkpoint_exists(model_path+model_name+".ckpd")):
                saver.restore(sess, model_path + model_name+".ckpd")
            else:
                print("check point " + model_path + model_name + ".ckpd not there")

        summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)

        if do_train:
            for epoch in range(num_training_epochs):
                inputs, supervisors = ld.getAWS1DataBatch(fcsv_names, str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, fsizes, batch_size, len_seq, len_pred)
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
                        xse, tse =  ld.getAWS1DataBatch(fcsv_names, str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, fsizes, batch_size, len_seq, len_pred, True)
                        calc_accuracy(output_op, xse, tse)

            xse, tse = ld.getAWS1DataBatch(fcsv_names, str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, fsizes, batch_size, len_seq, len_pred, True)
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

            sdata = [[] for i in range(dim_vec_out)]
            pdata = [[] for i in range(dim_vec_out)]

            while(ifile < len(fcsv_names)):
                xse, tse = ld.getAWS1BatchSectionSeq(str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, len_pred, aws1_data, ipos, isection * 0.25, min(1.0, (isection + 1) * 0.25))
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
            header = "t"
            for key in str_vec_out:
                header = header+","+key
            
            print(header)
            for i in range(0, len_pred):
                record = ("%d" % i)
                for j in range(dim_vec_out):
                    record = record + (",%f" % loss[i][j])
                
                print(record)

sys.exit()
