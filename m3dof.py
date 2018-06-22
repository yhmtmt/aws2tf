import tensorflow as tf
import random
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import pdb
import csv
import ld
pdb.set_trace()

# 2 inputs [eng,rev]
# 1 outputs [rev]
dim_vec_in = 2
dim_vec_out = 1
len_pred = 32
length_of_sequences = 128
batch_size = 128
num_training_epochs = 20000
learning_rate = 0.001
forget_bias = 0.8

# mass=3000
# dt: time step (e.g. 0.132sec)


# Gear/Throttle Model
# eng: engine control value
# gnmax, gnmin: gear neutral range [gnmax, gnmin]
# tcu: time clutch up (e.g. 2.0sec)
# tcd: time clutch down (e.g. 2.0sec)
# rtu: rate throttle up (e.g. 1.0/3.0 rate/sec)
# rtd: rate throttle down (e.g. 1.0/3.0 rate/sec)
# at: throttle absorption (e.g. 0.05 (5%))
# ta: actual throttle position with absorption
# cf: final clutch position
# tf: final throttle position 
# c: clutch position
# t: throttle position
# str: {nf, fn, nb, bn, fu, fd, bu, bd}
# strp: previous transition
# state: {b, n, f}
#
# for [strp, str]
# case [fu, fu] or [bu, bu]
#       tp<-t
#       t <-t + dt rtu
#       ta<-t
# case [fu, fd] or [bu, bd]
#       tp<-t
#       t <-t - dt rtu
#       ta<-t
# case [fd, fu] or [bd, bu]
# case [fd, fd] or [bd, bd]

# Rev Model
# rev: rev value  
# rrev: rev rate
# revf: final rev value
# 
# update with dt: rev = (revf - rev) rrev dt




do_train = True
refine_existing_training = True
model_path = "/home/ubuntu/matumoto/model/"
model_name = "m3dof.ckpd"

print("Model:"+ model_path + model_name)


def inference(input_ph):
    with tf.name_scope("inference") as scope:
 #       vmass = tf.constant(value=mass,shape=[1],name="mass")       
  #      vmass_matrix = tf.get_variable("mass_matrix",[3,3],initializer=tf.zeros_initializer, trainable=False)
   #     vamass_matrix = tf.get_variable("added_mass_matrix",[3,3], initializer=tf.zeros_initializer, trainable=False)

        output =[]
        return output

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
    input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, dim_vec_in], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, dim_vec_out * len_pred], name="supervisor")

    output_op = inference(input_ph)
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
            if(tf.train.checkpoint_exists(model_path+model_name)):
                saver.restore(sess, model_path + model_name)
            else:
                print("check point " + model_path + model_name + " not there")

        summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)

        if do_train:
            for epoch in range(num_training_epochs):
                inputs, supervisors = ld.getAWS1DataBatch(fcsv_names, fsizes, batch_size, length_of_sequences, len_pred)
                train_dict = {
                    input_ph:      inputs,
                    supervisor_ph: supervisors,
                }
                train_loss, _ = sess.run([loss_op, training_op], feed_dict=train_dict)

                if (epoch) % 100 == 0:
                    summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                    print("train#%d, train loss: %e" % (epoch, train_loss))
                    summary_writer.add_summary(summary_str, epoch)
                    if (epoch) % 500 == 0:
                        xse, tse =  ld.getAWS1DataBatch(fcsv_names, fsizes, batch_size, length_of_sequences, len_pred, True)
                        calc_accuracy(output_op, xse, tse)

            xse, tse = ld.getAWS1DataBatch(fcsv_names, fsizes, batch_size, length_of_sequences, len_pred, True)
            calc_accuracy(output_op, xse, tse, prints=True)
            saver.save(sess, model_path + model_name)
        else:
            ifile=0
            isection=1
            ipos=0
            loss = np.array([0.] * dim_vec_out * len_pred)
            loss_total= 0.
            count=0
            print("Loading %s" % fcsv_names[ifile])
            t,eng,rud,rev,roll,rroll,pitch,rpitch,yaw,ryaw,cog,rcog,sog,rsog = ld.loadAWS1Data(fcsv_names[ifile]) 
            while(ifile < len(fcsv_names)):
                xse, tse = ld.getAWS1BatchSectionSeq(len_pred, eng, rud, rev, roll, rroll, pitch, rpitch, yaw, ryaw, cog, rcog, sog, rsog, ipos, isection * 0.25, min(1.0, (isection + 1) * 0.25))
                if(xse is None and tse is None):
                    if(isection == 1):
                        isection = 3
                    else:
                        ifile += 1
                        print("%d samples" % count)                        
                        if(ifile == len(fcsv_names)):
                            break
                        print("Loading %s" % fcsv_names[ifile])
                        t,eng,rud,rev,roll,rroll,pitch,rpitch,yaw,ryaw,cog,rcog,sog,rsog = ld.loadAWS1Data(fcsv_names[ifile]) 
                        isection = 1
                    ipos = 0
                    continue

                pred_dict = {
                    input_ph:  xse,
                }
                output = sess.run([output_op], feed_dict=pred_dict)
                abserr = abs(output - tse)
                loss += abserr[0][0]
                loss_total += np.mean(abserr)
                count += 1
                ipos += 1

            loss /= float(count)
            loss = np.reshape(loss, [len_pred, 6])
            loss_total /= float(count)

            print("loss_total %f" % loss_total)
            print("t, rroll, rpitch, ryaw, rcog, sog, rsog")
            for i in range(0, len_pred):
                print("%d,%f,%f,%f,%f,%f,%f" % (i+1, loss[i][0], loss[i][1], loss[i][2], loss[i][3], loss[i][4], loss[i][5]))

sys.exit()
