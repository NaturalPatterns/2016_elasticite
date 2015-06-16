#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Particle-like simulations using pyglet.app

Exploration mode.

    Interaction keyboard:
    - TAB pour passer/sortir du fulscreen
    - espace : passage en first-person perspective
    - H : more players
    - C : change viewpoint (cycle from one VP to another)

    Les interactions visuo - sonores sont simulées ici par des switches lançant des phases:
    - R : rugosité physique G_struct distance_struct
    - G : glissement = perceptif G_rot <> G_rot_hot
    - N : restore la config sans event = Neutre
    et des événements:
    - P : pulse (modif de la longueur et raideur des segments)
    - V : G_repulsion <> G_repulsion_hot
    - B : break 2&3
    - J : break 1
    - D : Down

"""
# HACK to avoid the "AccessInit: hash collision: 3 for both 1 and 1" bug...
# see http://forum.jetbrains.com/thread/PyCharm-938
import sys
import PIL.Image
sys.modules['Image'] = PIL.Image
# TODO: modele 3D blending fog / épaisseur du triangle / projection fond de la salle
# TODO: paramètre scan pour rechercher des bifurcations (edge of chaos)
# TODO: contrôle de la vitesse du mouvement de position simulé
########################################
from parametres import sliders, VPs, volume, p, kinects_network_config, d_x, d_y, d_z, scenario, calibration, DEBUG
from modele_dynamique import Scenario
s = Scenario(p['N'], scenario, volume, VPs, p, calibration)
########################################
do_firstperson, foc_fp, s_VP_fp, alpha_fp, int_fp, intB_fp, show_VP = False, 60., 0, .1, 1., 0.01, True
s.heading_fp, s.rot_heading_fp, s.inc_heading_fp = 0., 0., 0.1
s_VP = 0  # VP utilisé comme projecteur en mode projection
do_slider = True
do_slider = False
do_fs = not do_slider
do_sock = True
do_sock = False
do_interference = True
do_interference = False
########################################
i_win = 0
foc_VP = 50.
foc_VP = VPs[i_win]['foc']
n_players = 1
########################################
if do_sock:
    sys.path.append('../network/')
    from network import Kinects
    k = Kinects(kinects_network_config)
else:
    positions = None

# Window information
# ------------------
import pyglet
platform = pyglet.window.get_platform()
print "platform" , platform
display = platform.get_default_display()
print "display" , display
screens = display.get_screens()
print "screens" , screens
for i, screen in enumerate(screens):
    print 'Screen %d: %dx%d at (%d,%d)' % (i, screen.width, screen.height, screen.x, screen.y)
N_screen = len(screens) # number of screens
N_screen = 1# len(screens) # number of screens
assert N_screen == 1 # we should be running on one screen only

from pyglet.window import Window

if do_fs:
    win_0 = Window(screen=screens[0], fullscreen=True, resizable=True)
else:
    win_0 = Window(width=screen.width*2/3, height=screen.height*2/3, screen=screens[0], fullscreen=False, resizable=True)
    win_0.set_location(screen.width/3, screen.height/3)

import pyglet.gl as gl
fps_text = pyglet.clock.ClockDisplay()
from pyglet.gl.glu import gluLookAt
import numpy as np
def on_resize(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glEnable(gl.GL_BLEND)
    gl.glShadeModel(gl.GL_SMOOTH) #
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)# gl.GL_FASTEST)# gl.GL_NICEST)# GL_DONT_CARE)#
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDisable(gl.GL_LINE_SMOOTH)
    gl.glColor3f(1.0, 1.0, 1.0)
    gl.glDisable(gl.GL_CLIP_PLANE0)
    gl.glDisable(gl.GL_CLIP_PLANE1)
    gl.glDisable(gl.GL_CLIP_PLANE2)
    gl.glDisable(gl.GL_CLIP_PLANE3)
    return pyglet.event.EVENT_HANDLED

win_0.on_resize = on_resize
win_0.set_visible(True)
gl.glMatrixMode(gl.GL_MODELVIEW)
gl.glLoadIdentity()
gl.gluPerspective(foc_VP, 1.0*win_0.width/win_0.height, VPs[i_win]['pc_min'], VPs[i_win]['pc_max'])
gluLookAt(VPs[i_win]['x'], VPs[i_win]['y'], VPs[i_win]['z'],
      VPs[i_win]['cx'], VPs[i_win]['cy'], VPs[i_win]['cz'],
      0., 0, 1.0)

events = [0, 0, 0, 0, 0, 0, 0, 0] # 8 types d'événéments

@win_0.event
def on_key_press(symbol, modifiers):
    global events, do_firstperson, s, s_VP, s_VP_fp, n_players
    if symbol == pyglet.window.key.TAB:
        if win_0.fullscreen:
            win_0.set_fullscreen(False)
            win_0.set_location(screen.width/3, screen.height/3)
        else:
            win_0.set_fullscreen(True)
    elif symbol == pyglet.window.key.SPACE:
        do_firstperson = not(do_firstperson)
    elif symbol == pyglet.window.key.LEFT:
        s.rot_heading_fp += s.inc_heading_fp
    elif symbol == pyglet.window.key.RIGHT:
        s.rot_heading_fp -= s.inc_heading_fp
    elif symbol == pyglet.window.key.N:
        events = [0, 0, 0, 0, 0, 0, 0, 0]
    elif symbol == pyglet.window.key.R:
        events = [1, 0, 0, 0, 0, 0, 0, 0]
        #events[0] = 1 - events[0]
    elif symbol == pyglet.window.key.G:
        events = [0, 0, 0, 0, 1, 0, 0, 0]
        #events[4] = 1 - events[4]
    elif symbol == pyglet.window.key.B:
        events = [1, 1, 1, 1, 1, 1, 1, 1] # break 1
    elif symbol == pyglet.window.key.J:
        events = [1, 1, 1, 1, 1, 1, 1, 0] # break 2
    elif symbol == pyglet.window.key.O:
        events = [1, 1, 1, 1, 1, 1, 0, 0] # break 3
    elif symbol == pyglet.window.key.P:
        events[1] = 1 - events[1]
    elif symbol == pyglet.window.key.V:
        events[2] = 1 - events[2]
    elif symbol == pyglet.window.key.S:
        events[7] = 1 - events[7]
    elif symbol == pyglet.window.key.H:
        n_players = (n_players + 1) %5
        print 'n_players: ', n_players
    elif symbol == pyglet.window.key.C:
        s_VP = (s_VP + 1) % s.nvps
        s_VP_fp = (s_VP_fp + 1) % s.nvps
        print 'you are sitting in the eye of VP no ', s_VP
    else:
        print symbol
        print events

from numpy import sin, cos, pi

@win_0.event
def on_resize(width, height):
    print 'The window was resized to %dx%d' % (width, height)
@win_0.event
def on_draw():
    global s, s_VP, s_VP_fp, n_players, t1, t0
    t = s.t

    if do_sock:
        positions = k.read_sock() #
    else:
        # pour simuler ROGER:
        amp, amp2 = .1, .3
        T, T2 = 25., 30. # periode en secondes
        positions_ = []
        positions_.append([s.roger[0], s.roger[1], s.roger[2]]) #  bouge pas, roger.
        positions_.append([s.roger[0] * (1. + amp*cos(2*pi*s.t/T2)), s.roger[1] * (.8 + amp*cos(2*pi*s.t/T)), 1.*s.roger[2]]) # une autre personne dans un mouvement en phase
        positions_.append([s.roger[0] * (1. + amp*sin(2*pi*s.t/T2)), s.roger[1] * (1.2 + amp*sin(2*pi*s.t/T)), 1.2*s.roger[2]]) # une autre personne dans un mouvement en phase
        positions_.append([s.roger[0] * (1. + amp2*cos(2*pi*s.t/T2)), s.roger[1] * (1. + amp2*cos(2*pi*s.t/T)), .5*s.roger[2]]) # une autre personne dans un mouvement en phase
        positions_.append([s.roger[0], s.roger[1] * (1. + amp2*cos(2*pi*s.t/T2)), .9*s.roger[2]]) # une personne dans un mouvement circulaire (elipse)
        positions = []
        for position in positions_[:n_players]:
            positions.append(position)

         #[[9.71, 1.5, 1.03], [9.17, 3.61, 1.17], [10.98, 4.43, 1.85], [7.68, 4.53, 1.68]]

    s.do_scenario(positions=positions, events=events)
    #if DEBUG: print  s.particles[0:3, :].mean(axis=1), s.particles[3:6, :].mean(axis=1), s.particles[0:3, :].std(axis=1), s.particles[3:6, :].std(axis=1)

    win_0.clear()
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    if do_firstperson:
        gl.glEnable(gl.GL_FOG)
        gl.glFogi (gl.GL_FOG_MODE, gl.GL_LINEAR)
        # gl.glFogfv (gl.GL_FOG_COLOR, [0.8,0.8,0.8, 1.])
        gl.glHint (gl.GL_FOG_HINT, gl.GL_NICEST)#GL_DONT_CARE)
        gl.glFogf (gl.GL_FOG_DENSITY, 0.0000001)
        gl.glFogf (gl.GL_FOG_START, .0)
        gl.glFogf (gl.GL_FOG_END, 60.0)
        # gl.glClearColor(0.5, 0.5, 0.5, 1.0)

        gl.gluPerspective(foc_fp, 1.0*win_0.width/win_0.height,
                          VPs[s_VP_fp]['pc_min'], VPs[s_VP_fp]['pc_max'])
        #x_fp, y_fp, z_fp = positions[0][0], positions[0][1], positions[0][2]
        x_fp, y_fp, z_fp = s.croix
        s.heading_fp += s.rot_heading_fp * (s.t -t) # 2* pi * s.t / 30
        gluLookAt(x_fp, y_fp, z_fp,
                  x_fp + np.cos(s.heading_fp), y_fp + np.sin(s.heading_fp), z_fp,
                  0., 0, 1.0)
        # montre la salle comme un joli parallélipède bleu
        #gl.glPointSize(10)
        #gl.glColor3f(0., 0., 1.)
        #salle = [[0., 0., 0., d_x, 0., 0.], [0., 0., 0., 0., d_y, 0.], [0., 0., 0., 0., 0., d_z]]
        #pyglet.graphics.draw(2*3, gl.GL_LINES, ('v3f', salle))¬
        for i_VP, VP in enumerate(VPs):
            # marque la postion de chaque VP par un joli carré vert
            if show_VP:
                gl.glPointSize(10)
                gl.glColor3f(0., 1., 0.)
                pyglet.graphics.draw(1, gl.GL_POINTS, ('v3f', [VP['x'], VP['y'], VP['z']]))

            VP_ = np.array([[VP['x'], VP['y'], VP['z']]]).T * np.ones((1, s.N))
            p_ = s.particles[0:6, i_VP*s.N:(i_VP+1)*s.N].copy()
            # projecting the segment on the wall opposite to the VPs
            if VP['x'] > d_x/2: # un VP du coté x=0, on projete sur le plan x=0
                p_[1] = d_x / (d_x - p_[0]) * (p_[1]-VP['y']) + VP['y']
                p_[2] = d_x / (d_x - p_[0]) * (p_[2]-VP['z']) + VP['z']
                p_[4] = d_x / (d_x - p_[3]) * (p_[4]-VP['y']) + VP['y']
                p_[5] = d_x / (d_x - p_[3]) * (p_[5]-VP['z']) + VP['z']
                p_[0] = 0
                p_[3] = 0
            else:
                p_[1] = d_x / (p_[0]) * (p_[1]-VP['y']) + VP['y']
                p_[2] = d_x / (p_[0]) * (p_[2]-VP['z']) + VP['z']
                p_[4] = d_x / (p_[3]) * (p_[4]-VP['y']) + VP['y']
                p_[5] = d_x / (p_[3]) * (p_[5]-VP['z']) + VP['z']
                p_[0] = d_x
                p_[3] = d_x
            colors_ = np.array([int_fp, int_fp, int_fp, alpha_fp, intB_fp, intB_fp, intB_fp, alpha_fp, intB_fp, intB_fp, intB_fp, alpha_fp])[:, np.newaxis] * np.ones((1, s.N))
            pyglet.graphics.draw(3*s.N, gl.GL_TRIANGLES,
                                 ('v3f', np.vstack((VP_, p_)).T.ravel().tolist()),
                                 ('c4f', colors_.T.ravel().tolist()))

    else:
        gl.glDisable(gl.GL_FOG)
        #gl.glMatrixMode(gl.GL_PROJECTION)
        #gl.glLoadIdentity()
        gl.gluPerspective(foc_VP, 1.0*win_0.width/win_0.height,
                          VPs[s_VP]['pc_min'], VPs[s_VP]['pc_max'])
        gluLookAt(VPs[s_VP]['x'], VPs[s_VP]['y'], VPs[s_VP]['z'],
                  VPs[s_VP]['cx'], VPs[s_VP]['cy'], VPs[s_VP]['cz'],
                  0., 0, 1.0)
        #gl.glMatrixMode(gl.GL_MODELVIEW)
        #gl.glLoadIdentity()

        # TODO: make an option to view particles from above
        #gl.gluOrtho2D(0.0, d_y, 0., d_z) #left, right, bottom, top, near, far

        gl.glLineWidth (p['line_width'])
        # marque la postion des personnes par un joli carré rouge
        for position in positions:
            gl.glPointSize(10)
            gl.glColor3f(1., 0., 0.)
            pyglet.graphics.draw(1, gl.GL_POINTS, ('v3f', position))

        gl.glColor3f(1., 1., 1.)

        pyglet.graphics.draw(2*s.N, gl.GL_LINES, ('v3f', s.particles[0:6, s_VP*s.N:(s_VP+1)*s.N].T.ravel().tolist()))

    if do_sock: k.trigger()

def callback(dt):
    global do_sock
    try :
        if DEBUG:
            pass
            #print '%f seconds since last callback' % dt , '%f  fps' % pyglet.clock.get_fps()
    except :
        pass

if s.scenario=='leapfrog' and do_slider:
    fig = sliders(s.p)

pyglet.clock.schedule(callback)
pyglet.app.run()
print 'Goodbye'
