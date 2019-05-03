# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:31:14 2018

@author: jieqiang (jieqiang.wei@gmail.com)

The tracking part of the code is based on "Geometric tracking control of a quadrotor UAV on SE(3)". As a source of
uncertainty, we assume that the mass used in the controller is different from the actual mass. This could be due to
measurement noise, or even mass drop during flight. We identify the model uncertainty using Gaussian process and Neural
network.
"""
"""This code has no numerical approximations, but everything is calculated analytically."""


import numpy as np
#import quaternion
from scipy.integrate import solve_ivp
from numpy.linalg import norm as no
from functools import reduce
import mat_fun as mf
import time


class Drone(object):
    def __init__(self, m_true, m_controller):
        self.mass_sys = m_true
        self.mass_ctrl = m_controller
        self.gravity = 9.81
        self.inertia = np.diag([0.0820,0.0845,0.1377])

        # 0-2 position,3-5 velocity,6-14 rotation matrix,15-17 ang_velocity
        #self.state = np.array[self.p, self.v.tolist(), self.r.tolist(), self.w.tolist()]
        self.state = np.array([0., 0., 0.,0., 0., 0.,1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])

        self.t_prev = 0

        self.Omega_d_prev = np.array([0.,0.,0.])


    def parse_new_states(self, new_states):
        self.state = new_states
        # print self.state


    def update(self, t_0, t_1, t_step):
        # the return of solve_ivp has ret.t (time points), ret.y (values of solution at time points)
        # t_eval defines at what time points the values of the solution are stored, default one is automatically chosen by the solver
        # self.t_current += dt
        # t_eval = np.linspace(self.t_prev,self.t_current, 2)

        t_eval = np.linspace(t_0, t_1, int((t_1-t_0)/t_step))  # add t_eval to parse for plotting
        initial_state = self.state
        ret = solve_ivp(self.drone_dyna, [t_0,t_1], initial_state, method='RK45', t_eval=t_eval)

        #self.parse_new_states(ret.y[:,-1].flatten())
        return ret.t, ret.y



    def differential_x_d(self,t):
        # calculations based on x_d which is the desired trajectory

        x_d = np.array([0.4 * t, 0.4 * np.sin(np.pi * t), 0.6 * np.cos(np.pi * t)])
        v_d = np.array([0.4, 0.4 * np.pi * np.cos(np.pi * t), -0.6 * np.pi * np.sin(np.pi * t)])
        x_ddotdot = np.array([0, -0.4 * np.pi ** 2 * np.sin(np.pi * t), -0.6 * np.pi ** 2 * np.cos(np.pi * t)])
        x_ddotdotdot = np.array([0, -0.4 * np.pi ** 3 * np.cos(np.pi * t), 0.6 * np.pi ** 3 * np.sin(np.pi * t)])
        x_ddotdotdotdot = np.array([0, 0.4 * np.pi ** 4 * np.sin(np.pi * t), 0.6 * np.pi ** 4 * np.cos(np.pi * t)])
        return x_d,v_d,x_ddotdot,x_ddotdotdot,x_ddotdotdotdot

    def differential_b_1d(self,t):
        # the desired direction b_1d
        b_1d = np.array([np.cos(np.pi * t), np.sin(np.pi * t), 0])
        b_1ddot = np.array([-np.pi * np.sin(np.pi * t), np.pi * np.cos(np.pi * t), 0])
        b_1ddotdot = np.array([- (np.pi **2) * np.cos(np.pi * t), - (np.pi**2) * np.sin(np.pi * t), 0])
        return b_1d,b_1ddot,b_1ddotdot


    def controller(self, t, y):


        dt = max(t-self.t_prev,0.01)#no if statement for code speed up.

        # dt = t-self.t_prev
        # if dt < 0.01:
        #     dt = 0.01
        #print "dt",dt

        self.t_prev = t

        p = y[0:3]
        v = y[3:6]
        R = y[6:15].reshape(3, 3, order='F')
        Omega = y[15:18]
        Omega_hat = mf.w_hat(Omega)


        # calculation based on the desired direction b_1d
        b_1d,b_1ddot,b_1ddotdot = self.differential_b_1d(t)

        # calculation based on the desired trajectory x_d
        x_d,v_d,x_ddotdot,x_ddotdotdot,x_ddotdotdotdot = self.differential_x_d(t)

        # the errors: e_x e_v
        e_x = p - x_d
        e_v = v - v_d

        # to calculate b_3d
        e_3 = np.array([0., 0., 1.])
        b_3d_temp = -k_x * e_x - k_v * e_v - self.mass_ctrl * self.gravity * e_3 + self.mass_ctrl * x_ddotdot
        b_3d = -b_3d_temp / no(b_3d_temp)

        # control input f
        f = - np.dot(b_3d_temp, np.dot(R, e_3))

        # construct R_d
        b_2d_temp = np.cross(b_3d_temp, b_1d)  # this is different from the paper
        b_2d = b_2d_temp / no(b_2d_temp)

        R_d = reduce(np.append, [np.cross(b_2d, b_3d), b_2d, b_3d]).reshape(3, 3, order='F')

        # calculate \dot{R}_d: R_ddot
        b_3d_temp_dot = -k_x * e_v - k_v * (
                self.gravity * e_3 - f * np.dot(R, e_3) / self.mass_ctrl - x_ddotdot) + self.mass_ctrl * x_ddotdotdot
        b_2d_temp_dot = np.cross(b_3d_temp_dot, b_1d) + np.cross(b_3d_temp, b_1ddot)
        b_3ddot = -mf._derive_normed_vector(b_3d_temp,b_3d_temp_dot)
        b_2ddot = mf._derive_normed_vector(b_2d_temp,b_2d_temp_dot)
        R_ddot = reduce(np.append, [np.cross(b_2ddot, b_3d) + np.cross(b_2d, b_3ddot), b_2ddot, b_3ddot]).reshape(3, 3, order='F')

        # calculate Omega_d
        Omega_d_hat = np.dot(R_d.transpose(), R_ddot)
        Omega_d = mf.matrix_hat_inv(Omega_d_hat)


        # calculate e_R, e_Omega
        e_R = 0.5 * mf.matrix_hat_inv( np.dot(R_d.transpose(), R) - np.dot(R.transpose(), R_d) )
        e_Omega = Omega - reduce(np.dot, [R.transpose(), R_d, Omega_d])


        # calculate Omega_ddot numerically
        #Omega_ddot = (Omega_d-self.Omega_d_prev)/dt

        #self.Omega_d_prev = Omega_d

        # calculate Omega_ddot analytically
        ddv = - (np.dot(b_3d_temp,reduce(np.dot,[R,Omega_hat, e_3])) * np.dot(R,e_3)
                #reduce(np.dot,[R,Omega_hat, e_3]) * np.dot(b_3d_temp, np.dot(R,e_3))
                 + np.dot(R,e_3) * np.dot(b_3d_temp_dot, np.dot(R,e_3))
                 + np.dot(R,e_3) * np.dot(b_3d_temp, reduce(np.dot, [R,Omega_hat, e_3])))/self.mass_ctrl
        b_3d_temp_dotdot = (- k_x * (self.gravity * e_3 - f * np.dot(R, e_3) / self.mass_ctrl - x_ddotdot)
                            - k_v * (ddv - x_ddotdotdot)  + self.mass_ctrl * x_ddotdotdotdot)

        b_3ddotdot = -mf._derive_derived_normed_vector(b_3d_temp,b_3d_temp_dot,b_3d_temp_dotdot)

        b_2d_temp_dotdot = np.cross(b_3d_temp_dotdot, b_1d) + np.cross(b_3d_temp_dot, b_1ddot) \
                           + np.cross(b_3d_temp_dot, b_1ddot) + np.cross(b_3d_temp, b_1ddotdot)
        b_2ddotdot = mf._derive_derived_normed_vector(b_2d_temp,b_2d_temp_dot,b_2d_temp_dotdot)

        Rd_1ddotdot = np.cross(b_2ddotdot, b_3d) + np.cross(b_2ddot, b_3ddot) + np.cross(b_2ddot, b_3ddot) + np.cross(b_2d, b_3ddotdot)

        R_ddotdot = reduce(np.append, [Rd_1ddotdot,b_2ddotdot,b_3ddotdot]).reshape(3, 3, order='F')

        Omega_ddot = mf.matrix_hat_inv(np.dot(R_d.transpose(),(R_ddotdot-np.dot(R_ddot,Omega_d_hat))))


        # control input
        tau = -k_R*e_R - k_Omega * e_Omega + reduce(np.dot,[Omega_hat,self.inertia,Omega])\
              - np.dot(self.inertia,(reduce(np.dot,[Omega_hat,R.transpose(),R_d,Omega_d])-reduce(np.dot,[R.transpose(),R_d,Omega_ddot])))


        return f, tau, Omega_d


    def drone_dyna(self, t, y):

        #global  p,v,R,Omega,Omega_hat

        #v = y[3:6]
        #R = y[6:15].reshape(3, 3, order='F')


        e_3 = np.array([0., 0., 1.])
        #Omega = y[15:18]


        Omega_hat = mf.w_hat(y[15:18])

        f,tau, _ = self.controller(t, y)


        # initialization
        state_dot = np.zeros(18)
        # velocity
        state_dot[0:3] = y[3:6]
        # acceleration
        state_dot[3:6] = np.array([0., 0., self.gravity] - f*np.dot(y[6:15].reshape(3, 3, order='F'), e_3)/self.mass_sys)
        state_dot[6:15] = np.dot(y[6:15].reshape(3, 3, order='F'), Omega_hat).reshape(9, order='F')
        state_dot[15:18] = np.dot(np.linalg.inv(self.inertia), np.dot(np.dot(-Omega_hat, self.inertia), y[15:18]) + tau)

        return state_dot




import numpy as np


t_start = 0
t_stop = 20
t_step = 0.05
el = int((t_stop-t_start)/t_step)
t_list = np.linspace(t_start, t_stop, el)


drone = Drone(m_true=4.34,m_controller=4.34)

# the control gains
global k_x,k_v,k_R,k_Omega
k_x = 16.
k_v = 5.6
k_R = 8.81
k_Omega = 2.54


pos_x = np.array([])
pos_y = np.array([])
pos_z = np.array([])
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

time_update, state_update = drone.update(t_start,t_stop,t_step) # without specific sampling time points

end = time.time()
print "The running time is:", end - start

for i in xrange(len(time_update)):
    pos_x = np.append(pos_x, state_update[0,i])
    pos_y = np.append(pos_y, state_update[1,i])
    pos_z = np.append(pos_z, state_update[2,i])
    w_x = np.append(w_x, state_update[15,i])
    w_y = np.append(w_y, state_update[16,i])
    w_z = np.append(w_z, state_update[17,i])
    pos_d0 = np.append(pos_d0, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[0])
    pos_d1 = np.append(pos_d1, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[1])
    pos_d2 = np.append(pos_d2, np.array([0.4 * time_update[i], 0.4 * np.sin(np.pi * time_update[i]), 0.6 * np.cos(np.pi * time_update[i])])[2])
    _,_,temp =drone.controller(time_update[i],state_update[:,i])
    w_d0 = np.append(w_d0, temp[0])
    w_d1 = np.append(w_d1, temp[1])
    w_d2 = np.append(w_d2, temp[2])

# print "dimensions: ",t_list.shape,pos_x.shape,pos_d0.shape, w_d0.shape



""" the plots zone """

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plot_method  = raw_input("Which kind of plot do you want to see, 2D or 3D?")

if plot_method == '3D':
    mpl.rcParams['legend.fontsize'] = 11
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pos_d0,pos_d1,pos_d2,label='desired trajectory')
    ax1.plot(pos_x,pos_y,pos_z,label='drone position')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(w_d0,w_d1,w_d2,label='desired angular velocity')
    ax2.plot(w_x,w_y,w_z,label='drone angular velocity')
    ax2.legend()
elif plot_method == '2D':
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
    print "plot method has to be 3D or 2D!"

plt.show()

"""Data collection part"""

"""ML part"""