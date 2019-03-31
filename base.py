# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:31:14 2018

@author: jieqiang (jieqiang.wei@gmail.com)

This is the code based on paper "Geometric tracking control of a quadrotor UAV on SE(3)". with some machine learning extension.
"""
import numpy as np
#import quaternion
from scipy.integrate import solve_ivp

class Drone(object):
    def __init__(self, m):
        self.mass = m
        self.gravity = -9.8
        self.inertia = np.diag([1.,1.,1.])
        self.r = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
        self.p = np.array([0., 0., 0.])
        self.v = np.array([0., 0., 0.])
        self.w = np.array([0., 0., 0.])
        # 0-2 position,3-5 velocity,6-14 rotation matrix,15-17 ang_velocity
        #self.state = np.array[self.p, self.v.tolist(), self.r.tolist(), self.w.tolist()]
        self.state = np.array([0., 0., 0.,0., 0., 0.,1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])

        print (self.state)

        #self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')


    def rotation_matrix(self):
        #q = np.quaternion(self.state[7:11])
        #R = quaternion.as_rotation_matrix(q)
        R = self.state[6:15].reshape(3,3)
        return R


    def w_hat_create(self, w):
        omega_hat = np.array([0.,-w[2],w[1],w[2],0,-w[0],-w[1],w[0],0.]).reshape(3,3)
        return omega_hat


    def update(self, dt):

        ret = solve_ivp(self.drone_dyna, [0,dt], self.state, method='RK45', t_eval=[dt])
        print ret
        self.parse_new_states(ret.y.flatten())
        print type(ret.y.flatten())

    def parse_new_states(self, new_states):
        self.p = new_states[0:3]
        self.v = new_states[3:6]
        self.r = new_states[6:15]
        self.w = new_states[15:18]

        self.state = new_states

    def controller(self, r):

        return F, tau


    def drone_dyna(self, t, y):  # , t , d_12, d_23, d_31):
        # control input
        F = np.zeros(3)
        tau = np.zeros(3)



        p = y[0:3]
        v = y[3:6]
        R = y[6:15].reshape(3, 3,order='F')
        omega = y[15:18]
        omega_hat = self.w_hat_create(omega)

        # initialization
        state_dot = np.zeros(18)
        # velocity
        state_dot[0:3] = v
        # acceleration
        state_dot[3:6] = np.array([0., 0., self.gravity] + np.dot(R, F))
        state_dot[6:15] = np.dot(R, omega_hat).reshape(9, )
        state_dot[15:18] = np.dot(np.linalg.inv(self.inertia), np.dot(np.dot(-omega_hat, self.inertia), omega) + tau)
        return state_dot




import numpy as np
#from scipy.integrate import solve_ivp as ode
import matplotlib.pyplot as plt

# b,c,d are three edge lengths of the desired formation
b = 3.0
c = 4.0
d = 5.0


t_start = 0
t_stop = 10
dt = 0.1
el = int((t_stop-t_start)/dt)
t_list = np.linspace(t_start, t_stop, el)

drone = Drone(m=1)

drone.init_att = np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.])
drone.init_pos = np.array([0.,0.,0.])
drone.init_vel = np.array([0.,0.,0.])
drone.init_att_vel = np.array([0.,0.,0.])


for t in t_list:
    print t
    drone.update(dt)

