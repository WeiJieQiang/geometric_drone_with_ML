# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:31:14 2018

@author: jieqiang (jieqiang.wei@gmail.com)

This is the code based on paper "Geometric tracking control of a quadrotor UAV on SE(3)". with some machine learning extension.
"""
import numpy as np
#import quaternion
from scipy.integrate import solve_ivp
from numpy import linalg as LA
from functools import reduce
import mat_fun as mf

class Drone(object):
    def __init__(self, m):
        self.mass = m
        self.gravity = 9.81
        self.inertia = np.diag([0.0820,0.0845,0.1377])
        self.r = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
        self.p = np.array([0., 0., 0.])
        self.v = np.array([0., 0., 0.])
        self.w = np.array([0., 0., 0.])
        # 0-2 position,3-5 velocity,6-14 rotation matrix,15-17 ang_velocity
        #self.state = np.array[self.p, self.v.tolist(), self.r.tolist(), self.w.tolist()]
        self.state = np.array([0., 0., 0.,0., 0., 0.,1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])

        self.t_prev = 0.0
        self.t_current = 0.

        self.Omega_d_current = np.array([0.,0.,0.])
        self.Omega_d_prev = np.array([0.,0.,0.])
        self.Omega_d_d = np.array([0.,0.,0.])



    # def rotation_matrix(self):
    #     #q = np.quaternion(self.state[7:11])
    #     #R = quaternion.as_rotation_matrix(q)
    #     rot = self.state[6:15].reshape(3,3)
    #     return rot

    def parse_new_states(self, new_states):
        self.p = new_states[0:3]
        self.v = new_states[3:6]
        self.r = new_states[6:15]
        self.w = new_states[15:18]

        self.state = new_states
        # print self.state


    def update(self, dt):
        # the return of solve_ivp has ret.t (time points), ret.y (values of solution at time points)
        # t_eval defines at what time points the values of the solution are stored, default one is automatically chosen by the solver
        self.t_current += dt
        #print "current time:", self.t_current
        t_eval = np.linspace(0, dt, 2)
        ret = solve_ivp(self.drone_dyna, [self.t_prev,self.t_current], self.state, method='RK45')#, t_eval=t_eval)
        #print "the time points:",ret.t
        #print "the time points are ", ret.t
        #print ret.y.shape
        #print ret
        # self.t_prev = self.t_current - ret.t[-1]
        # _,_,self.Omega_d_prev = self.controller(self.t_current-dt,self.state)
        self.parse_new_states(ret.y[:,-1].flatten())
        # #print "prev time:", self.t_prev
        # _,_,self.Omega_d_current = self.controller(self.t_current,self.state)
        # if self.t_current-self.t_prev > 0.00001:
        #     self.Omega_d_d = (self.Omega_d_current-self.Omega_d_prev)/(self.t_current-self.t_prev)
        # else:
        #     print "time is too short!"

        #print "the stats omega_d_prev,current:", self.Omega_d_prev,self.Omega_d_current
        #print "the time diff:", self.t_current-self.t_prev
        #print type(ret.y.flatten())




    def controller(self, t, y):

        dt = t-self.t_prev

        if dt < 0.00001:
            dt = 0.01

        self.t_prev = t

        p = y[0:3]
        v = y[3:6]
        R = y[6:15].reshape(3, 3, order='F')
        Omega = y[15:18]
        Omega_hat = mf.w_hat(Omega)

        # the control gains
        k_x = 16.
        k_v = 5.6
        k_R = 8.81
        k_Omega = 2.54

        # the desired trajectory x_d and desired direction b_1d
        x_d = np.array([0.4 * t, 0.4 * np.sin(np.pi * t), 0.6 * np.cos(np.pi * t)])
        b_1d = np.array([np.cos(np.pi * t), np.sin(np.pi * t), 0])
        b_1ddot = np.array([-np.pi * np.sin(np.pi * t), np.pi * np.cos(np.pi * t), 0])

        # calculations based on x_d
        v_d = np.array([0.4, 0.4 * np.pi * np.cos(np.pi * t), -0.6 * np.pi * np.sin(np.pi * t)])
        x_ddotdot = np.array([0, -0.4 * np.pi ** 2 * np.sin(np.pi * t), -0.6 * np.pi ** 2 * np.cos(np.pi * t)])
        x_ddotdotdot = np.array([0, -0.4 * np.pi ** 3 * np.cos(np.pi * t), 0.6 * np.pi ** 3 * np.sin(np.pi * t)])

        # the errors: e_x e_v
        e_x = p - x_d
        e_v = v - v_d

        # to calculate b_3d
        global e_3
        e_3 = np.array([0., 0., 1.])
        b_3d_temp = -k_x * e_x - k_v * e_v - self.mass * self.gravity * e_3 + self.mass * x_ddotdot
        if LA.norm(b_3d_temp)> 0.0001:
            b_3d = - b_3d_temp / LA.norm(b_3d_temp)
        else:
            print "The norm:", LA.norm(b_3d_temp)

        # control input f
        f = - np.dot(b_3d_temp, np.dot(R, e_3))

        # construct R_d
        b_2d_temp = np.cross(b_3d_temp, b_1d)  # this is different from the paper
        b_2d = b_2d_temp / LA.norm(b_2d_temp)

        R_d = reduce(np.append, [np.cross(b_2d, b_3d), b_2d, b_3d]).reshape(3, 3,
                                                                            order='F')  ### Q1: this one should be F, but why the others??? Or it just needs to be consistent
        # print "cross", np.cross(b_2d, b_3d)
        # print "b_2d", b_2d
        # print "b_3d", b_3d
        # print "R_d", R_d
        # calculate \dot{R}_d: R_ddot
        b_3d_temp_dot = -k_x * e_v - k_v * (
                self.gravity * e_3 - f * np.dot(R, e_3) / self.mass - x_ddotdot) + self.mass * x_ddotdotdot
        b_2d_temp_dot = np.cross(b_3d_temp_dot, b_1d) + np.cross(b_3d_temp, b_1ddot)
        b_3ddot = LA.norm(b_3d_temp) ** (-3) * np.dot(np.dot(b_3d_temp.reshape(3, 1), b_3d_temp.reshape(1, 3))
                                                      - LA.norm(b_3d_temp) ** 2 * np.identity(3), b_3d_temp_dot)
        b_2ddot = LA.norm(b_2d_temp) ** (-3) * np.dot(-np.dot(b_2d_temp.reshape(3, 1), b_2d_temp.reshape(1, 3))
                                                      + LA.norm(b_2d_temp) ** 2 * np.identity(3), b_2d_temp_dot)
        R_ddot = reduce(np.append, [np.cross(b_2ddot, b_3d) + np.cross(b_2d, b_3ddot), b_2ddot, b_3ddot]).reshape(3, 3,
                                                                                                                  order='F')

        # calculate Omega_d
        Omega_d_hat = np.dot(R_d.transpose(), R_ddot)
        # print "the Omega_d_hat:", Omega_d_hat
        Omega_d = mf.matrix_hat_inv(Omega_d_hat)


        # calculate e_R
        e_R = 0.5 * mf.matrix_hat_inv( np.dot(R_d.transpose(), R) - np.dot(R.transpose(), R_d) )

        #print "e_R", e_R

        # calculate e_Omega
        e_Omega = Omega - reduce(np.dot, [R.transpose(), R_d, Omega_d])

        # calculate Omega_ddot

        #print "controller times:",t


        Omega_ddot = (Omega_d-self.Omega_d_prev)/dt

        self.Omega_d_prev = Omega_d

        #Omega_ddot = self.Omega_d_d # outside the callback function.
        #print Omega_ddot
        # if t > 0.0:
        #     Omega_ddot = (Omega_d - self.Omega_d_prev)/t # inside the callback
        # else:
        #     Omega_ddot = np.array([0.,0.,0.])

        # control input
        tau = -k_R*e_R - k_Omega * e_Omega + reduce(np.dot,[Omega_hat,self.inertia,Omega])\
              - np.dot(self.inertia,(reduce(np.dot,[Omega_hat,R.transpose(),R_d,Omega_d])-reduce(np.dot,[R.transpose(),R_d,Omega_ddot])))


        return f, tau, Omega_d


    def drone_dyna(self, t, y):

        #global  p,v,R,Omega,Omega_hat

        v = y[3:6]
        R = y[6:15].reshape(3, 3, order='F')

        #print "R:",R
        e_3 = np.array([0., 0., 1.])
        #print "np.dot(R, e_3):",np.dot(R, e_3)
        Omega = y[15:18]


        Omega_hat = mf.w_hat(Omega)

        f,tau,_ = self.controller(t, y)


        # initialization
        state_dot = np.zeros(18)
        # velocity
        state_dot[0:3] = v
        #print "sudu:", v
        # acceleration
        state_dot[3:6] = np.array([0., 0., self.gravity] - f*np.dot(R, e_3)/self.mass)
        #print "the acce:",np.array([0., 0., self.gravity] - f*np.dot(R, e_3)/self.mass)
        state_dot[6:15] = np.dot(R, Omega_hat).reshape(9, order='F')
        state_dot[15:18] = np.dot(np.linalg.inv(self.inertia), np.dot(np.dot(-Omega_hat, self.inertia), Omega) + tau)
        return state_dot




import numpy as np
#from scipy.integrate import solve_ivp as ode
import matplotlib.pyplot as plt



t_start = 0
t_stop = 4
dt = 0.05
el = int((t_stop-t_start)/dt)
t_list = np.linspace(t_start, t_stop, el)


#the dimension of interest
d=0

drone = Drone(m=4.34)
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

for t_iter in t_list:
    # print t
    drone.update(dt)
    pos_x = np.append(pos_x, drone.state[0])
    pos_y = np.append(pos_y, drone.state[1])
    pos_z = np.append(pos_z, drone.state[2])
    w_x = np.append(w_x, drone.state[15])
    w_y = np.append(w_y, drone.state[16])
    w_z = np.append(w_z, drone.state[17])
    pos_d0 = np.append(pos_d0, np.array([0.4 * t_iter, 0.4 * np.sin(np.pi * t_iter), 0.6 * np.cos(np.pi * t_iter)])[0])
    pos_d1 = np.append(pos_d1, np.array([0.4 * t_iter, 0.4 * np.sin(np.pi * t_iter), 0.6 * np.cos(np.pi * t_iter)])[1])
    pos_d2 = np.append(pos_d2, np.array([0.4 * t_iter, 0.4 * np.sin(np.pi * t_iter), 0.6 * np.cos(np.pi * t_iter)])[2])
    w_d0 = np.append(w_d0,drone.Omega_d_current[0])
    w_d1 = np.append(w_d1, drone.Omega_d_current[1])
    w_d2 = np.append(w_d2, drone.Omega_d_current[2])

print "dimensions: ",t_list.shape,pos_x.shape,pos_d0.shape, w_d0.shape
import matplotlib.pyplot as plt

plt.plot(t_list,pos_x, 'r')
plt.plot(t_list,pos_y, 'g')
plt.plot(t_list,pos_z, 'b')
plt.plot(t_list,pos_d0, 'k')
plt.plot(t_list,pos_d1, 'm')
plt.plot(t_list,pos_d2, 'y')
# plt.plot(t_list,w_d0, 'r')
# plt.plot(t_list,w_d1,'g')
# plt.plot(t_list,w_d2, 'b')
# plt.plot(t_list,w_x, 'k')
# plt.plot(t_list,w_y, 'm')
# plt.plot(t_list,w_z, 'y')
plt.show()