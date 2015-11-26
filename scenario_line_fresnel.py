#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Sur une ligne de lames, on fait tourner les lames avec un mouvement relativement élastique mais exogene (prédéterminé, pas émergent)

"""

import elasticite as el
import numpy as np

duration = el.get_default_args(el.EdgeGrid.render)['duration']
location = el.get_default_args(el.EdgeGrid.render)['location']

class EdgeGrid(el.EdgeGrid):
    def update(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame
        xf = location[0] #+ .2 * np.sin(2*np.pi*(self.t)/duration)
        zf = location[2] + 4. * np.cos(2*np.pi*(self.t)/duration)
        self.lames[2, :N_lame] = np.mod(np.pi/2 + np.arctan2(self.lames[1, :N_lame]-zf, self.lames[0, :N_lame]-xf), np.pi)

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'

    e = EdgeGrid(N_lame=25, grid_type='line', mode=mode, verb=False)
    el.main(e)
