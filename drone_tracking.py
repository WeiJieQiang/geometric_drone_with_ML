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
        # the control gains
        global k_x, k_v, k_R, k_Omega
        k_x = 16.
        k_v = 5.6
        k_R = 8.81
        k_Omega = 2.54


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