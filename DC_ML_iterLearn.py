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

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t_start = 0
t_stop = 20
t_step = 0.05
el = int((t_stop-t_start)/t_step)
t_list = np.linspace(t_start, t_stop, el)


drone_c = Drone(m_true=4.34,m_controller=3)

time_update, state_update = drone_c.update(t_start,t_stop,t_step) # without specific sampling time points


#position, velocity, attitude, angular velocity
[pos_x,pos_y,pos_z,v_1,v_2,v_3,R_1,R_2,R_3,R_4,R_5,R_6,R_7,R_8,R_9,w_x,w_y,w_z] = state_update


pos_d0 = np.array([])
pos_d1 = np.array([])
pos_d2 = np.array([])

w_d0 = np.array([])
w_d1 = np.array([])
w_d2 = np.array([])


for i in xrange(len(time_update)):
    pos_d0 = np.append(pos_d0, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[0])
    pos_d1 = np.append(pos_d1, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[1])
    pos_d2 = np.append(pos_d2, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[2])
    _,_,temp =drone_c.controller(time_update[i],state_update[:,i])
    w_d0 = np.append(w_d0, temp[0])
    w_d1 = np.append(w_d1, temp[1])
    w_d2 = np.append(w_d2, temp[2])



""" the plots zone """

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

plt.show()


"""Data collection part"""

"""S1: total state with imperfect controller, stored in state_imperfect_ctrl"""
state_imperfect_ctrl = state_update

"""S2: state_dot with perfect controller, stored in state_dot_perfect_ctrl. Q: how to generate this in reality is a big question! 
A possible answer is that this perfect controller is runned in a simulation."""
#drone_i = Drone(m_true=4.34,m_controller=4.34)
drone_i = Drone(m_true=3,m_controller=3)

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
#print "test:", state_dot_imperfect_ctrl[:,50]-state_dot_perfect_ctrl[:,50]
#print "test:", Y.shape

#print "########## Data info:############"
#print "We have %d samples" % Y.shape[1]
#print "The input dimension is %d" % (state_imperfect_ctrl.shape[0]+control_imperfect_ctrl.shape[0])
#print "The target dimension is %d" % Y.shape[0]

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
    m.optimize('bfgs', messages=True, max_f_eval = 1000, max_iters=2e3)
    np.save('./model_save.npy', m.param_array)



# fig = m.plot()
# display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
# display(m)

# validation on GPy
# print"EEE2"
# print X.shape
# print X[:,0].shape
# dim = 200
# print m.predict(X[:,dim].reshape(1,22))[0]-Y[:,dim]
#
# temp =np.zeros(len(Y[1]))
# for i in xrange(len(Y[1])):
#     temp[i] = no(m.predict(X[:,i].reshape(1,22))[0]-Y[:,i])
#
# ti = range(400)
# plt.plot(ti,temp)
# #plt.show()

# iteratively run the drone with regression corrections
n=2 # number of iteration of drone running
for run_num in xrange(n):

    start = time.time()
    """collect data and append it to X and Y"""
    drone_reg = Drone_reg(m_true=4.34, m_controller=3)
    time_update, state_update = drone_reg.update(t_start, t_stop, t_step)  # without specific sampling time points

    """S1: total state with imperfect controller, stored in state_imperfect_ctrl"""
    state_imperfect_ctrl = state_update

    """S2: state_dot with perfect controller, stored in state_dot_perfect_ctrl. Q: how to generate this in reality is a big question! 
    A possible answer is that this perfect controller is runned in a simulation."""
    #drone_i = Drone(m_true=4.34, m_controller=4.34)
    drone_i = Drone(m_true=3, m_controller=3)

    #print state_imperfect_ctrl[:, 1].shape

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

    """ML part: here we use some libraries to perform the regression task for the model uncertainty. GPy, scikit-learn, tensorflow+keras"""


    # The sample data X
    X_temp = np.zeros(shape=(22, len(time_update)))
    X_temp[0:18, :] = state_imperfect_ctrl
    X_temp[18:22, :] = control_imperfect_ctrl

    X = np.append(X,X_temp, axis=1)


    np.save('./X.npy', X)
    np.save('./Y.npy', Y)



    k = GPy.kern.RBF(input_dim=22, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X.transpose(), Y.transpose(), k)#Q: when use Sparse, then the load needs to be changed.
    m.constrain_positive('')
    m.optimize('bfgs', messages=True, max_f_eval = 1000, max_iters=2e1)
    np.save('./model_save.npy', m.param_array)

    print "The iteration %d is done." %run_num
    end = time.time()
    print "The running time is:", end - start

"""the final run of drone"""
drone_reg = Drone_reg(m_true=4.34,m_controller=3)

# plot the corrected trajectory using regression model
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

[pos_x_reg,pos_y_reg,pos_z_reg,v_1_reg,v_2_reg,v_3_reg,R_1_reg,R_2_reg,
 R_3_reg,R_4_reg,R_5_reg,R_6_reg,R_7_reg,R_8_reg,R_9_reg,w_x_reg,w_y_reg,w_z_reg] = state_update_reg

for i in xrange(len(time_update_reg)):
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