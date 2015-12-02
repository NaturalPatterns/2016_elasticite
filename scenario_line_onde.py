#!/usr/bin/env python
# -*- coding: utf8 -*-

import elasticite as el
import numpy as np

def make_vague(impulse=False):
    import os
    import numpy as np
    import MotionClouds as mc

    mc.N_X, mc.N_Y, mc.N_frame = 50, 2, 2048
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    theta, B_theta = 0., np.pi/8.
    alpha, sf_0, B_sf, B_V = 1., .02, .1, .1
    seed = 1234565
    V_X, V_Y = .1, 0.
    mc_wave = mc.envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y, B_V=B_V, 
                                theta=theta, B_theta=B_theta, sf_0=sf_0, B_sf=B_sf, alpha=alpha)

    wave = mc.random_cloud(mc_wave, seed=seed, impulse=impulse)
    
    wave -= wave.mean()
    wave /= np.abs(wave).max()
    return wave

class EdgeGrid(el.EdgeGrid):
    def __init__(self, vague, x_offset=0, y_offset=0, t_offset=0, N_steps = 2048, damp_tau=5., **kwargs):
        #super(el.EdgeGrid.__init__(self))
        #super(el.EdgeGrid, self).__init__(**kwargs)
        el.EdgeGrid.__init__(self, **kwargs)
        #print (self.verb, kwargs)
        self.vague = vague
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.t_offset = t_offset
        self.N_steps = N_steps
        #print(self.x_offset, self.y_offset, self.t_offset)
        #print(self.z.shape)
        self.damp_tau = damp_tau
        
        
    def update(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame
        damp = lambda t: 1. - np.exp(-np.abs(np.mod(t+self.period/2, self.period)-self.period/2)/self.damp_tau)

        N_periods = 1
        i = np.mod(np.int(self.t/self.period * self.vague.shape[2] / N_periods), self.vague.shape[2])
        surface = np.zeros_like(self.lames[2, :N_lame])
        #for k, amp in zip([-2, -1, 0, 1, 2], [.125, .25, .5, .25, .125]):
        #    surface += amp * self.vague[self.x_offset:(self.x_offset+N_lame), self.y_offset, self.t_offset+i+k]
        surface = self.vague[self.x_offset:(self.x_offset+N_lame), self.y_offset, self.t_offset+i].ravel()
        #surface = np.convolve(surface, np.arange(5), mode='same')
        #dsurface = np.gradient(surface)
        #dsurface = np.roll(surface, 1) - surface
        #dsurface *= np.bartlett(N_lame)
        # print(dsurface.mean(), np.abs(dsurface).max(), damp(self.t))
        #dsurface /= np.abs(dsurface).max()
        surface *= np.tan(np.pi/12)*damp(self.t) # maximum angle achieved
        #print(surface.min(), surface.mean(), np.abs(surface).max(), damp(self.t))
        self.lames[2, :N_lame] = np.arctan(surface)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'
        
    vague_dense = make_vague(impulse=False)
    print (np.abs(vague_dense).max())

    period = 2048./30
    e = EdgeGrid(N_lame=25, grid_type='line', mode=mode,
                 verb=False, period=period, filename='mat/line_onde_dense.npy', 
                 vague = vague_dense,
                 x_offset=0, y_offset=0, t_offset=0, N_steps=2048)

    # running the code
    el.main(e)