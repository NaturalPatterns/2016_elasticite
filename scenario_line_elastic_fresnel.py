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
        force = np.zeros_like(self.lames[2, :])
        damp = lambda t: 0.01 #* np.exp(np.cos(t / 6.) / 3.**2)
        XF = lambda t: location[0]
        ZF = lambda t: location[2] + 3.5 * np.sin(2*np.pi*(self.t)/duration)
        xf = lambda t: XF(t)/self.total_width + .5
        zf = lambda t: ZF(t)/self.total_width + .5
        
        desired_angle = np.pi/2 + np.arctan2(self.lames[1, :]-zf(self.t), self.lames[0, :]-xf(self.t))
        self.lames[2, :] = np.mod(self.lames[2, :]-np.pi/2, np.pi) + np.pi/2
        force += np.mod(desired_angle-np.pi/2, np.pi) + np.pi/2- self.lames[2, :]
        force -= damp(self.t) * self.lames[3, :]/self.dt
        return 3. * force

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'

    e = EdgeGrid(N_lame=20, grid_type='line', mode=mode, verb=False)
    el.main(e)
