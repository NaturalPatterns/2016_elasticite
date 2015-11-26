#!/usr/bin/env python
# -*- coding: utf8 -*-

import elasticite as el
import numpy as np
import os

def master(e, filename):
    if e.structure: N_lame = e.N_lame-e.struct_N
    else: N_lame = e.N_lame

    def montage(z, z_in):
        z_in[:, 0] += z[-1, 0] + e.desired_fps # increment the time on the new array
        print (z.shape, z_in.shape)
        return np.vstack((z, z_in))

    burnout_time = 4.
    z = np.zeros((1, N_lame+1)) # zero at zero
    np.vstack((z, np.hstack((np.array(burnout_time), np.zeros(N_lame) ))))
    
    matpath = 'mat/'
    z_s = {}
    for scenario in ['line_vague_dense', 'line_vague_solo', 'line_fresnelastique']:
        z_s[scenario] = np.load(os.path.join(matpath, scenario + '.npy'))
    
    ###########################################################################
    z = montage(z, z_s['line_vague_dense'])
    ###########################################################################
    z = montage(z, z_s['line_vague_solo'])
    ###########################################################################
    z = montage(z, z_s['line_fresnelastique'])
    ###########################################################################
    z = montage(z, z_s['line_vague_dense'])
    
    ###########################################################################
    # save the file
    np.save(filename, z)

        

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'
        
    filename='mat/master.npy'
    
    e = el.EdgeGrid(N_lame=25, grid_type='line', mode=mode,
                 verb=True, filename=filename)

    master(e, filename)

    # running the code
    el.main(e)