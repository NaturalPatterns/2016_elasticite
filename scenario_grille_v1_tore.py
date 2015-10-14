#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Dans une grille en tore (pacman) privilégie les co-circulrités en évitant les co-linéarités

Short-range pour donner des patterns locaux

"""
import sys
if len(sys.argv)>1: mode = sys.argv[1]
else: mode = 'both'

import elasticite as el
import numpy as np
class EdgeGrid(el.EdgeGrid):
    def champ(self):
        force = np.zeros_like(self.lames[2, :])
        noise = lambda t: 1.* np.exp((np.cos(2*np.pi*(t-0.) / 6.)-1.)/ 1.5**2)
        damp = lambda t: 0.001 #* np.exp(np.cos(t / 6.) / 3.**2)
        cocir_t = lambda t: 1.5*np.exp((np.cos(2*np.pi*(t-2.) / 6.)-1.)/ .3**2)
        cocir_d = lambda d: np.exp(-d/.1)
        colin_t = lambda t: -2.*np.exp((np.cos(2*np.pi*(t-4.) / 6.)-1.)/ .5**2)
        colin_d = lambda d: np.exp(-d/.05)

        force += colin_t(self.t) * np.sum(np.sin(2*(self.angle_relatif()))*colin_d(self.distance(do_torus=True)), axis=1)
        force += cocir_t(self.t) * np.sum(np.sin(2*(self.angle_cocir(do_torus=True)))*cocir_d(self.distance(do_torus=True)), axis=1)
        force += noise(self.t)*np.pi*np.random.randn(self.N_lame)
        force -= damp(self.t) * self.lames[3, :]/self.dt
        speed = lambda t: 50.*(1-np.exp((np.cos(2*np.pi*(t-4.) / 6.)-1.)/ .5**2))
        return speed(self.t) * force

e = EdgeGrid(N_lame=16*72, mode=mode)
el.main(e)
