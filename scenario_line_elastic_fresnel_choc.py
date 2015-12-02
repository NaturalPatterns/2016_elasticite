#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Sur une ligne de lames, on fait tourner les lames avec un mouvement relativement élastique mais exogene (prédeterminé, pas émergent)

"""

import elasticite as el
import numpy as np

duration = el.get_default_args(el.EdgeGrid.render)['duration']
location = el.get_default_args(el.EdgeGrid.render)['location']


class EdgeGrid(el.EdgeGrid):
    def champ(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame

        force = np.zeros_like(self.lames[2, :N_lame])
        damp_min = 0.01
        damp_tau = 1.5
        damp_angle_tau = 8.
        damp_chirp_tau = 20.
        damp = lambda t: damp_min + (1.-damp_min)*np.exp(-np.abs(np.mod(t+self.period/2, self.period)-self.period/2)/damp_tau)
        damp_angle = lambda t: 1.-np.exp(-(np.mod(t+self.period/2, self.period)-self.period/2)**2/2/damp_angle_tau**2)
        step_angle = lambda t: 1.*(t < 3*self.period/4)
        xf = lambda t: location[0]
        freq = lambda t: 4 +  100*np.exp(-np.abs(t/damp_chirp_tau))
        chirp = lambda t: np.sin(2*np.pi*t/self.period * freq(t))
        amp = lambda t: 1 - np.exp(-np.abs(t/damp_chirp_tau))
        zf = lambda t: location[2] + 3.5 * amp(t) * chirp(t) * step_angle(t)

        smooth_tau = 15
        smooth = lambda t: 1.-np.exp(-np.abs(np.mod(t+self.period/2, self.period)-self.period/2)**2/smooth_tau**2)
        
        force -= 12 * (np.mod(self.lames[2, :N_lame]+np.pi/2, np.pi) - np.pi/2) * (1- smooth(self.t) )
        
        #print(freq(self.t), damp(self.t))
        desired_angle = np.arctan2(self.lames[1, :N_lame]-zf(self.t), self.lames[0, :N_lame]-xf(self.t)) - np.pi/2
        force += (damp_angle(self.t)*(np.mod(desired_angle+np.pi/2, np.pi) - np.pi/2) - self.lames[2, :N_lame]) *smooth(self.t)
        force -= damp(self.t) * self.lames[3, :N_lame]/self.dt
        return 3. * force

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'
    filename = None
    filename = 'mat/line_fresnelastique_choc.npy'
    e = EdgeGrid(N_lame=25, grid_type='line', mode=mode, verb=False, filename=filename, period=120.)
    el.main(e)

    