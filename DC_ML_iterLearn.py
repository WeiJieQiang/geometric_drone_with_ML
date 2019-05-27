"""
Created on Apr 14 15:31:14 2019

@author: jieqiang (jieqiang.wei@gmail.com)

This file perform the data collection and gaussian process regression for the mass uncertainty.
"""

"""This file has iterative data collection and learning to make 
the performance of the drone get better for each iteration."""

import numpy as np
from drone_tracking import *
#from drone_tracking_regression_correction import *
from numpy.linalg import norm as no
import GPy

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

"""S2: state_dot with perfect controller, stored in state_dot_perfect_ctrl. Q: how to generate this in reality is a big question! 
A possible answer is that this perfect controller is runned in a simulation."""
drone_i = Drone(m_true=4.34,m_controller=4.34)

print state_imperfect_ctrl[:,1].shape

state_dot_perfect_ctrl = np.zeros(shape = (18, len(time_update)))
for i in xrange(18):
    state_dot_perfect_ctrl[i] = [drone_i.drone_dyna(time_update[j],state_imperfect_ctrl[:,j])[i] for j in xrange(len(time_update))]

"""S3: state_dot and control signals with imperfect controller, stored in state_dot_imperfect_ctrl and control_imperfect_ctrl.
The first item is feasible by differentiate the trajectory of the real drone."""

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
# The sample data X
X = np.zeros(shape = (22, len(time_update)))
X[0:18,:] = state_imperfect_ctrl
X[18:22,:] = control_imperfect_ctrl

np.save('./X.npy', X)
np.save('./Y.npy', Y)

# GPy model training or reload
save_model = 0


if save_model:
    m = GPy.models.GPRegression(X.transpose(),Y.transpose())
    m.update_model(False)  # do not call the underlying expensive algebra on load
    m.initialize_parameter()  # Initialize the parameters (connect the parameters up)
    m[:] = np.load('model_save.npy')  # Load the parameters
    m.update_model(True)  # Call the algebra only once
else:
    k = GPy.kern.RBF(input_dim=22, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X.transpose(),Y.transpose(), k)
    m.constrain_positive('')
    m.optimize(messages=True,max_f_eval = 1000)
    print('test')
    np.save('./model_save.npy', m.param_array)


# fig = m.plot()
# display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
# display(m)

# validation on GPy
print"EEE2"
print X.shape
print X[:,0].shape
dim = 200
print m.predict(X[:,dim].reshape(1,22))[0]-Y[:,dim]

temp =np.zeros(len(Y[1]))
for i in xrange(len(Y[1])):
    temp[i] = no(m.predict(X[:,i].reshape(1,22))[0]-Y[:,i])

ti = range(400)
plt.plot(ti,temp)
#plt.show()

# iteratively run the drone with regression corrections
n=2 # number of iteration of drone running
for run_num in xrange(n):
    """collect data and append it to X and Y"""
    drone_reg = Drone_reg(m_true=4.34, m_controller=3)
    time_update, state_update = drone_reg.update(t_start, t_stop, t_step)  # without specific sampling time points

    # position
    pos_x = np.zeros(shape=(len(time_update),))
    pos_y = np.zeros(shape=(len(time_update),))
    pos_z = np.zeros(shape=(len(time_update),))

    # velocity
    v_1 = np.zeros(shape=(len(time_update),))
    v_2 = np.zeros(shape=(len(time_update),))
    v_3 = np.zeros(shape=(len(time_update),))

    # attitude
    R_1 = np.zeros(shape=(len(time_update),))
    R_2 = np.zeros(shape=(len(time_update),))
    R_3 = np.zeros(shape=(len(time_update),))
    R_4 = np.zeros(shape=(len(time_update),))
    R_5 = np.zeros(shape=(len(time_update),))
    R_6 = np.zeros(shape=(len(time_update),))
    R_7 = np.zeros(shape=(len(time_update),))
    R_8 = np.zeros(shape=(len(time_update),))
    R_9 = np.zeros(shape=(len(time_update),))

    # angular velocity
    w_x = np.zeros(shape=(len(time_update),))
    w_y = np.zeros(shape=(len(time_update),))
    w_z = np.zeros(shape=(len(time_update),))

    pos_d0 = np.zeros(shape=(len(time_update),))
    pos_d1 = np.zeros(shape=(len(time_update),))
    pos_d2 = np.zeros(shape=(len(time_update),))

    w_d0 = np.zeros(shape=(len(time_update),))
    w_d1 = np.zeros(shape=(len(time_update),))
    w_d2 = np.zeros(shape=(len(time_update),))

    for i in xrange(len(time_update)):
        pos_x[i] =  state_update[0, i]
        pos_y[i] = state_update[1, i]
        pos_z[i] = state_update[2, i]

        v_1[i] = state_update[3, i]
        v_2[i] = state_update[4, i]
        v_3[i] = state_update[5, i]

        R_1[i] = state_update[6, i]
        R_2[i] = state_update[7, i]
        R_3[i] = state_update[8, i]
        R_4[i] = state_update[9, i]
        R_5[i] = state_update[10, i]
        R_6[i] = state_update[11, i]
        R_7[i] = state_update[12, i]
        R_8[i] = state_update[13, i]
        R_9[i] = state_update[14, i]

        w_x[i] = state_update[15, i]
        w_y[i] = state_update[16, i]
        w_z[i] = state_update[17, i]

        pos_d0[i] = 0.4 * time_update[i]
        pos_d1[i] = 0.4 * np.sin(np.pi * time_update[i])
        pos_d2[i] = 0.6 * np.cos(np.pi * time_update[i])
        _, _, temp = drone_c.controller(time_update[i], state_update[:, i])
        w_d0[i] = temp[0]
        w_d1[i] = temp[1]
        w_d2[i] = temp[2]

    # print "dimensions: ",t_list.shape,pos_x.shape,pos_d0.shape, w_d0.shape

    """S1: total state with imperfect controller, stored in state_imperfect_ctrl"""

    state_imperfect_ctrl = np.zeros(shape=(18, len(time_update)))
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

    """S2: state_dot with perfect controller, stored in state_dot_perfect_ctrl. Q: how to generate this in reality is a big question! 
    A possible answer is that this perfect controller is runned in a simulation."""
    drone_i = Drone(m_true=4.34, m_controller=4.34)

    print state_imperfect_ctrl[:, 1].shape

    state_dot_perfect_ctrl = np.zeros(shape=(18, len(time_update)))
    for i in xrange(18):
        state_dot_perfect_ctrl[i] = [drone_i.drone_dyna(time_update[j], state_imperfect_ctrl[:, j])[i] for j in
                                     xrange(len(time_update))]

    """S3: state_dot and control signals with imperfect controller, stored in state_dot_imperfect_ctrl and control_imperfect_ctrl.
    The first item is feasible by differentiate the trajectory of the real drone."""

    state_dot_imperfect_ctrl = np.zeros(shape=(18, len(time_update)))
    control_imperfect_ctrl = np.zeros(shape=(4, len(time_update)))
    for i in xrange(18):
        state_dot_imperfect_ctrl[i] = [drone_reg.drone_dyna_without_prediction(time_update[j], state_imperfect_ctrl[:, j])[i] for j in
                                       xrange(len(time_update))]
    for j in xrange(len(time_update)):
        f_temp, tau_temp, _ = drone_reg.controller(time_update[j], state_imperfect_ctrl[:, j])
        control_imperfect_ctrl[0, j] = f_temp
        control_imperfect_ctrl[1:4, j] = tau_temp

    """S4: target variable, stored in Y"""
    Y = np.append(Y, state_dot_perfect_ctrl - state_dot_imperfect_ctrl, axis=1)

    """test if we have correct data collections"""
    #print "test:", state_dot_imperfect_ctrl[:, 50] - state_dot_perfect_ctrl[:, 50]
    print "test:", Y.shape

    print "########## Data info:############"
    print "We have %d samples" % Y.shape[1]
    print "The input dimension is %d" % (state_imperfect_ctrl.shape[0] + control_imperfect_ctrl.shape[0])
    print "The target dimension is %d" % Y.shape[0]

    """ML part: here we use some libraries to perform the regression task for the model uncertainty. GPy, scikit-learn, tensorflow+keras"""


    # The sample data X
    X_temp = np.zeros(shape=(22, len(time_update)))
    X_temp[0:18, :] = state_imperfect_ctrl
    X_temp[18:22, :] = control_imperfect_ctrl

    X = np.append(X,X_temp, axis=1)

    np.save('./X.npy', X)
    np.save('./Y.npy', Y)

    k = GPy.kern.RBF(input_dim=22, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X.transpose(), Y.transpose(), k)
    m.constrain_positive('')
    m.optimize(messages=True, max_f_eval=1000)
    print('test')
    np.save('./model_save.npy', m.param_array)

    print "The iteration %d is done." %run_num


# the final run of drone
drone_reg = Drone_reg(m_true=4.34,m_controller=3)

# plot the corrected trajectory using regression model
#position
pos_x_reg = np.array([])
pos_y_reg = np.array([])
pos_z_reg = np.array([])

# velocity
v_1_reg = np.array([])
v_2_reg = np.array([])
v_3_reg = np.array([])

# attitude
R_1_reg = np.array([])
R_2_reg = np.array([])
R_3_reg = np.array([])
R_4_reg = np.array([])
R_5_reg = np.array([])
R_6_reg = np.array([])
R_7_reg = np.array([])
R_8_reg = np.array([])
R_9_reg = np.array([])

# angular velocity
w_x_reg = np.array([])
w_y_reg = np.array([])
w_z_reg = np.array([])



pos_d0_reg = np.array([])
pos_d1_reg = np.array([])
pos_d2_reg = np.array([])

w_d0_reg = np.array([])
w_d1_reg = np.array([])
w_d2_reg = np.array([])

start = time.time()

time_update_reg, state_update_reg = drone_reg.update(t_start,t_stop,t_step) # without specific sampling time points

end = time.time()
print "The running time is:", end - start


for i in xrange(len(time_update_reg)):
    pos_x_reg = np.append(pos_x_reg, state_update_reg[0,i])
    pos_y_reg = np.append(pos_y_reg, state_update_reg[1,i])
    pos_z_reg = np.append(pos_z_reg, state_update_reg[2,i])

    v_1_reg = np.append(v_1_reg, state_update_reg[3, i])
    v_2_reg = np.append(v_2_reg, state_update_reg[4, i])
    v_3_reg = np.append(v_3_reg, state_update_reg[5, i])

    R_1_reg = np.append(R_1_reg, state_update_reg[6, i])
    R_2_reg = np.append(R_2_reg, state_update_reg[7, i])
    R_3_reg = np.append(R_3_reg, state_update_reg[8, i])
    R_4_reg = np.append(R_4_reg, state_update_reg[9, i])
    R_5_reg = np.append(R_5_reg, state_update_reg[10, i])
    R_6_reg = np.append(R_6_reg, state_update_reg[11, i])
    R_7_reg = np.append(R_7_reg, state_update_reg[12, i])
    R_8_reg = np.append(R_8_reg, state_update_reg[13, i])
    R_9_reg = np.append(R_9_reg, state_update_reg[14, i])

    w_x_reg = np.append(w_x_reg, state_update_reg[15,i])
    w_y_reg = np.append(w_y_reg, state_update_reg[16,i])
    w_z_reg = np.append(w_z_reg, state_update_reg[17,i])

    pos_d0_reg = np.append(pos_d0_reg, np.array([0.4 * time_update_reg[i], 0.4 * np.sin(np.pi * time_update_reg[i]), 0.6 * np.cos(np.pi * time_update_reg[i])])[0])
    pos_d1_reg = np.append(pos_d1_reg, np.array([0.4 * time_update_reg[i], 0.4 * np.sin(np.pi * time_update_reg[i]), 0.6 * np.cos(np.pi * time_update_reg[i])])[1])
    pos_d2_reg = np.append(pos_d2_reg, np.array([0.4 * time_update_reg[i], 0.4 * np.sin(np.pi * time_update_reg[i]), 0.6 * np.cos(np.pi * time_update_reg[i])])[2])
    _,_,temp =drone_reg.controller(time_update_reg[i],state_update_reg[:,i])
    w_d0_reg = np.append(w_d0_reg, temp[0])
    w_d1_reg = np.append(w_d1_reg, temp[1])
    w_d2_reg = np.append(w_d2_reg, temp[2])

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
    ax1.plot(pos_d0_reg,pos_d1_reg,pos_d2_reg,label='desired trajectory')
    ax1.plot(pos_x_reg,pos_y_reg,pos_z_reg,label='drone_c position')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(w_d0_reg,w_d1_reg,w_d2_reg,label='desired angular velocity')
    ax2.plot(w_x_reg,w_y_reg,w_z_reg,label='drone_c angular velocity')
    ax2.legend()
elif plot_method == '2':
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_update_reg, pos_x_reg, label='pos_x')
    ax1.plot(time_update_reg, pos_d0_reg, label='pos_d0')
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(time_update_reg, pos_y_reg, label='pos_y')
    ax2.plot(time_update_reg, pos_d1_reg, label='pos_d1')
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_update_reg, pos_z_reg, label='pos_z')
    ax3.plot(time_update_reg, pos_d2_reg, label='pos_d2')
    ax3.legend()

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_update_reg, w_x_reg, label='w_x')
    ax4.plot(time_update_reg, w_d0_reg, label='w_d0')
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_update_reg, w_y_reg, label='w_y')
    ax5.plot(time_update_reg, w_d1_reg, label='w_d1')
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(time_update_reg, w_z_reg, label='w_z')
    ax6.plot(time_update_reg, w_d2_reg, label='w_d2')
    ax6.legend()
else:
    print "plot method has to be 3 or 2!"

plt.show()