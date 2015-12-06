#!/usr/bin/env python
# -*- coding: utf8 -*-

import elasticite as el
import numpy as np
import os

def master(e, filename):
    if e.structure: N_lame = e.N_lame-e.struct_N
    else: N_lame = e.N_lame

    def montage(z, z_in, damp_tau=0.):
        z_out = z.copy()
        z_s = z_in.copy()
        #if damp_tau>0:
        #    max_time = z_in.shape[0]/e.desired_fps
        #    time = np.linspace(0., max_time, z_in.shape[0])
        #    smooth = 1.-np.exp((np.cos(2*np.pi* time / max_time)-1)/(damp_tau / max_time)**2)
        #    z_s[:, 1:] *= smooth[:, np.newaxis]

        #print (z_out[0, 0], z_out[-1, 0], z_s[0, 0], z_s[-1, 0])
        z_s[:, 0] += z_out[-1, 0] + 1./e.desired_fps # increment the time on the new array
        #print (z_out.shape, z_s.shape, z_s[0, 0], z_s[-1, 0])
        return np.vstack((z_out, z_s))

    def revert(z_in):
        z_s = z_in.copy()
        z_s[:, 1:] = z_s[:, 1:][:, ::-1]
        return z_s

    def mirror(z_in):
        z_s = z_in.copy()
        z_s[:, 1:] = -z_s[:, 1:]
        return z_s

    def interleave(z_1, z_2):
        z_s_1 = z_1.copy()
        z_s_2 = z_2.copy()
        z_s_1[:, 1::2] = z_s_2[:, 1::2]
        return z_s_1
            
    matpath = 'mat/'
    z_s = {}
    print('importing scenarii')
    for scenario in [#'line_vague_dense', 'line_vague_solo', 
                     'line_onde_dense', 'line_onde_solo', 'line_fresnelastique',
                    'line_fresnelastique_choc', 'line_fresnelastique_chirp', 
                     'line_geometry', 'line_geometry_45deg', 'line_geometry_90deg', 'line_geometry_structure']:
        z_s[scenario] = np.load(os.path.join(matpath, scenario + '.npy'))
        print(scenario)
        el.check(e, z_s[scenario])
    print('finished importing scenarii')    
    ###########################################################################
    burnout_time = 0.1
    z = np.zeros((1, N_lame+1)) # zero at zero
    z = np.vstack((z, np.hstack((np.array(burnout_time), np.zeros(N_lame) ))))
    ###########################################################################
#    z = montage(z, z_s['line_onde_solo'])
    z = montage(z, z_s['line_onde_dense'])
    ###########################################################################
    z = montage(z, z_s['line_geometry_90deg'])
    z = montage(z, z_s['line_geometry_45deg'])
    z = montage(z, z_s['line_geometry_structure'])
    z = montage(z, z_s['line_geometry'])
    z = montage(z, mirror(z_s['line_geometry_structure']))
    z = montage(z, mirror(z_s['line_geometry_45deg']))
    z = montage(z, mirror(z_s['line_geometry_90deg']))
    ###########################################################################
#    z = montage(z, z_s['line_onde_solo'])
    z = montage(z, revert(z_s['line_onde_dense']))
#    z = montage(z, revert(z_s['line_onde_solo']))
    ###########################################################################
    z = montage(z, z_s['line_geometry_structure'])
    z = montage(z, z_s['line_geometry'])
    z = montage(z, z_s['line_fresnelastique'])
    z = montage(z, mirror(z_s['line_fresnelastique']))
    z = montage(z, z_s['line_fresnelastique_chirp'])
    z = montage(z, z_s['line_fresnelastique_choc'])
    z = montage(z, z_s['line_geometry'])
    z = montage(z, z_s['line_geometry_structure'])
    ###########################################################################
    z = montage(z, z_s['line_fresnelastique'])
    z = montage(z, interleave(z_s['line_fresnelastique'], mirror(z_s['line_fresnelastique'])))
    z = montage(z, interleave(z_s['line_fresnelastique_chirp'], mirror(z_s['line_fresnelastique_choc'])))
    z = montage(z, interleave(z_s['line_fresnelastique_choc'], mirror(z_s['line_fresnelastique_chirp'])))
    z = montage(z, z_s['line_fresnelastique'])
    ###########################################################################
    z = montage(z, revert(z_s['line_onde_dense']))
#    z = montage(z, revert(z_s['line_onde_solo']))
    ###########################################################################
    # check that there is not overflow @ 30 fps
    el.check(e, z)
    ###########################################################################
    # save the file
    np.save(filename, z)

    return z_s

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1: mode = sys.argv[1]
    else: mode = 'both'
        
    filename = 'mat/master.npy'
    e = el.EdgeGrid(N_lame=25, grid_type='line', mode=mode,
                 verb=False, filename=filename)

    if mode == 'writer':
        z_s = master(e, filename)
    else:
        # running the code
        el.main(e)