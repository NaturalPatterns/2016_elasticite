#!/usr/bin/env python
# -*- coding: utf8 -*-

"""

Sur une ligne de lames, on fait tourner les lames avec un mouvement relativement élastique mais exogene (prédeterminé, pas émergent)

"""

import elasticite as el
import numpy as np

class EdgeGrid(el.EdgeGrid):
    def update(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame
        self.lames[2, :N_lame] = 90.* np.pi/180. * np.sin(2*np.pi*(self.t)/self.period)

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'

    #e = EdgeGrid(N_lame=25, grid_type='line', mode=mode, verb=False, period=30)
    # to test writing / reading from a file
    e = EdgeGrid(N_lame=25, grid_type='line', mode=mode, verb=True, period=30, filename='mat/line_contraint.npy')
    el.main(e)