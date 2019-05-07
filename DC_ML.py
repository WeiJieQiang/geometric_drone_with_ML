"""
Created on Apr 14 15:31:14 2019

@author: jieqiang (jieqiang.wei@gmail.com)

This file perform the data collection and gaussian process regression for the mass uncertainty.
"""

import numpy as np
from drone_tracking import *

t_start = 0
t_stop = 20
t_step = 0.05
el = int((t_stop-t_start)/t_step)
t_list = np.linspace(t_start, t_stop, el)


drone_c = Drone(m_true=4.34,m_controller=3)

#position
pos_x = np.array([])
pos_y = np.array([])
pos_z = np.array([])

# velocity
v_1 = np.array([])
v_2 = np.array([])
v_3 = np.array([])

# attitude
R_1 = np.array([])
R_2 = np.array([])
R_3 = np.array([])
R_4 = np.array([])
R_5 = np.array([])
R_6 = np.array([])
R_7 = np.array([])
R_8 = np.array([])
R_9 = np.array([])

# angular velocity
w_x = np.array([])
w_y = np.array([])
w_z = np.array([])



pos_d0 = np.array([])
pos_d1 = np.array([])
pos_d2 = np.array([])

w_d0 = np.array([])
w_d1 = np.array([])
w_d2 = np.array([])

start = time.time()

time_update, state_update = drone_c.update(t_start,t_stop,t_step) # without specific sampling time points

end = time.time()
print "The running time is:", end - start


for i in xrange(len(time_update)):
    pos_x = np.append(pos_x, state_update[0,i])
    pos_y = np.append(pos_y, state_update[1,i])
    pos_z = np.append(pos_z, state_update[2,i])

    v_1 = np.append(v_1, state_update[3, i])
    v_2 = np.append(v_2, state_update[4, i])
    v_3 = np.append(v_3, state_update[5, i])

    R_1 = np.append(R_1, state_update[6, i])
    R_2 = np.append(R_2, state_update[7, i])
    R_3 = np.append(R_3, state_update[8, i])
    R_4 = np.append(R_4, state_update[9, i])
    R_5 = np.append(R_5, state_update[10, i])
    R_6 = np.append(R_6, state_update[11, i])
    R_7 = np.append(R_7, state_update[12, i])
    R_8 = np.append(R_8, state_update[13, i])
    R_9 = np.append(R_9, state_update[14, i])

    w_x = np.append(w_x, state_update[15,i])
    w_y = np.append(w_y, state_update[16,i])
    w_z = np.append(w_z, state_update[17,i])

    pos_d0 = np.append(pos_d0, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[0])
    pos_d1 = np.append(pos_d1, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[1])
    pos_d2 = np.append(pos_d2, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[2])
    _,_,temp =drone_c.controller(time_update[i],state_update[:,i])
    w_d0 = np.append(w_d0, temp[0])
    w_d1 = np.append(w_d1, temp[1])
    w_d2 = np.append(w_d2, temp[2])

# print "dimensions: ",t_list.shape,pos_x.shape,pos_d0.shape, w_d0.shape


""" the plots zone """

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plot_method  = raw_input("Which kind of plot do you want to see, 2 or 3(D)?")

if plot_method == '3':
    mpl.rcParams['legend.fontsize'] = 11
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pos_d0,pos_d1,pos_d2,label='desired trajectory')
    ax1.plot(pos_x,pos_y,pos_z,label='drone_c position')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(w_d0,w_d1,w_d2,label='desired angular velocity')
    ax2.plot(w_x,w_y,w_z,label='drone_c angular velocity')
    ax2.legend()
elif plot_method == '2':
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_update, pos_x, label='pos_x')
    ax1.plot(time_update, pos_d0, label='pos_d0')
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(time_update, pos_y, label='pos_y')
    ax2.plot(time_update, pos_d1, label='pos_d1')
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_update, pos_z, label='pos_z')
    ax3.plot(time_update, pos_d2, label='pos_d2')
    ax3.legend()

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_update, w_x, label='w_x')
    ax4.plot(time_update, w_d0, label='w_d0')
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_update, w_y, label='w_y')
    ax5.plot(time_update, w_d1, label='w_d1')
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(time_update, w_z, label='w_z')
    ax6.plot(time_update, w_d2, label='w_d2')
    ax6.legend()
else:
    print "plot method has to be 3 or 2!"

#plt.show()

"""Data collection part"""

"""S1: total state with imperfect controller, stored in state_imperfect_ctrl"""

state_imperfect_ctrl = np.zeros(shape = (18,len(time_update)))
state_imperfect_ctrl[0] = pos_x
state_imperfect_ctrl[1] = pos_y
state_imperfect_ctrl[2] = pos_z
state_imperfect_ctrl[3] = v_1
state_imperfect_ctrl[4] = v_2
state_imperfect_ctrl[5] = v_3
state_imperfect_ctrl[6] = R_1
state_imperfect_ctrl[7] = R_2
state_imperfect_ctrl[8] = R_3
state_imperfect_ctrl[9] = R_4
state_imperfect_ctrl[10] = R_5
state_imperfect_ctrl[11] = R_6
state_imperfect_ctrl[12] = R_7
state_imperfect_ctrl[13] = R_8
state_imperfect_ctrl[14] = R_9
state_imperfect_ctrl[15] = w_x
state_imperfect_ctrl[16] = w_y
state_imperfect_ctrl[17] = w_z

"""S2: state_dot with perfect controller, stored in state_dot_perfect_ctrl. Q: how to generate this in reality is a big question!"""
drone_i = Drone(m_true=4.34,m_controller=4.34)

print state_imperfect_ctrl[:,1].shape

state_dot_perfect_ctrl = np.zeros(shape = (18, len(time_update)))
for i in xrange(18):
    state_dot_perfect_ctrl[i] = [drone_i.drone_dyna(time_update[j],state_imperfect_ctrl[:,j])[i] for j in xrange(len(time_update))]

"""S3: state_dot and control signals with imperfect controller, stored in state_dot_imperfect_ctrl and control_imperfect_ctrl"""

state_dot_imperfect_ctrl = np.zeros(shape = (18, len(time_update)))
control_imperfect_ctrl = np.zeros(shape = (4, len(time_update)))
for i in xrange(18):
    state_dot_imperfect_ctrl[i] = [drone_c.drone_dyna(time_update[j],state_imperfect_ctrl[:,j])[i] for j in xrange(len(time_update))]
for j in xrange(len(time_update)):
    f_temp,tau_temp,_ = drone_c.controller(time_update[j],state_imperfect_ctrl[:,j])
    control_imperfect_ctrl[0,j] = f_temp
    control_imperfect_ctrl[1:4,j] = tau_temp

"""S4: target variable, stored in Y"""
Y = state_dot_perfect_ctrl - state_dot_imperfect_ctrl

"""test if we have correct data collections"""
print "test:", state_dot_imperfect_ctrl[:,50]-state_dot_perfect_ctrl[:,50]
print "test:", Y.shape

print "########## Data info:############"
print "We have %d samples" % Y.shape[1]
print "The input dimension is %d" % (state_imperfect_ctrl.shape[0]+control_imperfect_ctrl.shape[0])
print "The target dimension is %d" % Y.shape[0]

"""ML part: here we use some libraries to perform the regression task for the model uncertainty. GPy, scikit-learn, tensorflow+keras"""
import GPy

# The sample data X
X = np.zeros(shape = (22, len(time_update)))
X[0:18,:] = state_imperfect_ctrl
X[18:22,:] = control_imperfect_ctrl

# GPy
# define kernel
ker = GPy.kern.RBF(input_dim=22, variance=1., lengthscale=1.)

# create simple GP model
m = GPy.models.GPRegression(X.transpose(),Y.transpose(),ker)

# optimize and plot
m.optimize(messages=True,max_f_eval = 1000)
# fig = m.plot()
# display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
# display(m)


"""feedback with regression of the uncertainty"""