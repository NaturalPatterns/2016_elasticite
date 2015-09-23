#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Dans une grille en tore (pacman) privilégie les co-linéarités à angles triangulaires.

On fait passer la contrainte par une vague spatiale exogene (prédeterminée, pas émergente)

"""
import sys
if len(sys.argv)>1: mode = sys.argv[1]
else: mode = 'both'

import elasticite as el
import numpy as np
class EdgeGrid(el.EdgeGrid, mode=mode):
    def champ(self):
        force = np.zeros_like(self.lames[2, :])
        noise = lambda t, x: 2* np.exp((np.cos(2*np.pi*((t-0.) / 6. + x))-1.)/ .1**2)
        damp = lambda t: 0.001 #* np.exp(np.cos(t / 6.) / 3.**2)
        colin_t = lambda t, y: -8.*np.exp((np.cos(2*np.pi*((t-3.) / 6. + y))-1.)/ .3**2)
        colin_d = lambda d: np.exp(-d/.05) #np.exp(-np.log((d+1.e-12)/.05)**2/2/1.5)

        #delta_angle = np.mod(self.angle_relatif()-np.pi/3., 2*np.pi/3)
        delta_angle = self.angle_relatif()-np.pi/3.
        #delta_angle *= np.sign(delta_angle)
        force += colin_t(self.t, self.lames[1, :]) * np.sum(np.sin(6*delta_angle)*colin_d(self.distance(do_torus=True)), axis=1)
        force += noise(self.t, self.lames[0, :])*np.pi*np.random.randn(self.N_lame)
        force -= damp(self.t) * self.lames[3, :]/self.dt
        return 100. * force

e = EdgeGrid()
el.main(e)
