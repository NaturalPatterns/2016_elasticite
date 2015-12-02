#!/usr/bin/env python
# -*- coding: utf8 -*-

import elasticite as el
import numpy as np

class EdgeGrid(el.EdgeGrid):
    def champ(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame

        force = np.zeros_like(self.lames[2, :N_lame])
        damp_min = 0.8
        damp_tau = 15.
        damp = lambda t: damp_min #+ (1.-damp_min)*np.exp(-np.abs(np.mod(t+self.period/2, self.period)-self.period/2)/damp_tau)

        smooth = lambda t: 1.-np.exp(-np.abs(np.mod(t+self.period/2, self.period)-self.period/2)**2/damp_tau**2)
        on_off = lambda t, freq: (np.sin(2*np.pi*t/self.period*freq) > 0.)

        noise = lambda t: .4 * smooth(t)
        np.random.seed(12345)
        random_timing = np.random.rand(N_lame)
        #print(np.mod(random_timing + self.t/self.period, 1))
        struct_angles = np.random.permutation( np.hstack((self.struct_angles, -np.array(self.struct_angles)))) * np.pi / 180
        #print(struct_angles, (np.mod(random_timing + self.t/self.period, 1)*len(struct_angles)).astype(np.int)) 
        angle_desired = np.zeros(N_lame)
        for i, idx in enumerate((np.mod(random_timing + self.t/self.period, 1)*len(struct_angles)).astype(np.int)):
            angle_desired[i] = struct_angles[idx]
        
        force -= 20 * (np.mod(self.lames[2, :N_lame] - angle_desired +np.pi/2, np.pi) - np.pi/2 ) * smooth(self.t)
        force -= 80 * (np.mod(self.lames[2, :N_lame] + np.pi/2, np.pi) - np.pi/2) * (1- smooth(self.t) )
        force += noise(self.t)*np.pi*np.random.randn(N_lame)
        force -= damp(self.t) * self.lames[3, :N_lame]/self.dt
        force = .02 * 100 * np.tanh(force/100)
        return force    

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'
    filename = None
    filename = 'mat/line_geometry_structure.npy'
    e = EdgeGrid(N_lame=25, grid_type='line', mode=mode, verb=False, filename=filename, period=4*60.)
    el.main(e)

    