#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import time

try:
	#import matplotlib
	#matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)
	import matplotlib.pyplot as plt
except:
	pass
#
# https://zeromq.github.io/pyzmq/serialization.html
def send_array(socket, A, flags=0, copy=True, track=False):
    import zmq
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    # buf = buffer(msg)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def mirror(particles, segment, alpha=1.):
    """
    Reflète les points ``particles`` par rapport à ``segment``.
    
    See 2015-11-02 élasticité expansion en miroir
    
    """
    
    mirror = particles.copy()
    perp = np.array([segment[1][1] - segment[1][0], -(segment[0][1] - segment[0][0])])
    d = perp[0]*(segment[0][1] - particles[0, :]) + perp[1]*(segment[1, 1] - particles[1, :])
    mirror[:2, :] =  particles[:2, :] + 2. * d[np.newaxis, :] * perp[:, np.newaxis] / (perp**2).sum()
    if mirror.shape[0]>2: 
        mirror[2, :] =  alpha
    return mirror


import inspect
def get_default_args(func):
    """
    
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))

# from  multiprocessing import Process

class EdgeGrid():
    def __init__(self,
                 N_lame = 8*72,
                 N_lame_X = None,
                 figsize = 13,
                 line_width = 4.,
                 grid_type = 'hex',
                 structure = True, struct_angles = [-15., -65., -102.],
                 verb = False,
                 mode = 'both',
                 filename = None,
                 period = None,
                 **kw_args
                 ):
        self.t0 = self.time(True)
        self.t = self.time()
        self.dt = self.t - self.t0
        self.verb = verb
        self.display = (mode=='display') or (mode=='both')
        self.stream =  (mode=='stream') or (mode=='display')
        #if mode=='display': self.stream = True
        self.filename = filename
        self.serial =  (mode=='serial') # converting a stream to the serial port to control the arduino
        if self.serial: self.verb=True
#         self.desired_fps = 750.
        self.desired_fps = 30.
        self.structure = structure
        self.screenshot = True # saves a screenshot after the rendering

        self.port = "5556"
        # moteur:
        self.serial_port, self.baud_rate = '/dev/ttyUSB0', 115200
        # 1.8 deg par pas (=200 pas par tour) x 32 divisions de pas
        # demultiplication : pignon1= 14 dents, pignon2 = 60 dents
        self.n_pas = 200. * 32. * 60 / 14
        # TODO : Vitesse Max moteur = 1 tour en 3,88
        self.n_pas_max = 140

        # taille installation
        self.total_width = 8 # en mètres
        self.lames_width = 5 # en mètres
        self.lames_height = 3 # en mètres
        self.background_depth = 100 # taille du 104 en profondeur
        self.f = .1
        self.struct_N = 6
        self.struct_position = [0., 3.5]
        self.struct_longueur = 3.
        self.struct_angles = struct_angles
        self.figsize = figsize
        self.line_width = line_width
        self.grid_type = grid_type
        self.grid(N_lame=N_lame, N_lame_X=N_lame_X)
        # self.lames[2, :] = np.pi*np.random.rand(self.N_lame)

        self.N_particles_per_lame = 2**3
        self.N_particles = self.struct_N * self.N_particles_per_lame
        if structure: self.sample_structure()

        # enregistrement / playback
        self.period = period
        self.load()

    def load(self):
        if not self.filename is None:
            if os.path.isfile(self.filename):
            # le fichier existe, on charge
                self.z = np.load(self.filename)
                self.period = self.z[:, 0].max()

    def time(self, init=False):
        if init: return time.time()
        else: return time.time() - self.t0

    def grid(self, N_lame, N_lame_X):
        """
        The coordinates of the screen are centered on the (0, 0) point and axis are the classical convention:

         y
         ^
         |
         +---> x

         angles are from the horizontal, then in trigonometric order (anticlockwise)

         """

        self.DEBUG = False
        self.DEBUG = True

        self.N_lame = N_lame
        #if N_lame_X is None:

        if self.grid_type=='hex':
            self.N_lame_X = np.int(np.sqrt(self.N_lame))#*np.sqrt(3) / 2)
            self.lames = np.zeros((4, self.N_lame))
            self.lames[0, :] = np.mod(np.arange(self.N_lame), self.N_lame_X)
            self.lames[0, :] += np.mod(np.floor(np.arange(self.N_lame)/self.N_lame_X), 2)/2
            self.lames[1, :] = np.floor(np.arange(self.N_lame)/self.N_lame_X)
            self.lames[1, :] *= np.sqrt(3) / 2
            self.lames[0, :] /= self.N_lame_X
            self.lames[1, :] /= self.N_lame_X
            self.lames[0, :] += .5/self.N_lame_X - .5
            self.lames[1, :] += 1.5/self.N_lame_X # TODO : prove analytically
            self.lames[0, :] *= self.total_width
            self.lames[1, :] *= self.total_width
            self.lame_length = .99/self.N_lame_X*self.total_width*np.ones(self.N_lame)
            self.lame_width = .03/self.N_lame_X*self.total_width*np.ones(self.N_lame)
        elif self.grid_type=='line':
            self.N_lame_X = self.N_lame
            self.lames = np.zeros((4, self.N_lame))
            self.lames[0, :] = np.linspace(-self.lames_width/2, self.lames_width/2, self.N_lame, endpoint=True)
            #self.lames[1, :] = self.total_width/2
            self.lame_length = .12*np.ones(self.N_lame) # en mètres
            self.lame_width = .042*np.ones(self.N_lame) # en mètres

        if self.structure: self.add_structure()

        self.lames_minmax = np.array([self.lames[0, :].min(), self.lames[0, :].max(), self.lames[1, :].min(), self.lames[1, :].max()])

    def do_structure(self):
        structure_ = np.zeros((3, self.struct_N))
        chain = np.zeros((2, 4))
        chain[:, 0] = np.array(self.struct_position).T
        for i, angle in enumerate(self.struct_angles):
            chain[0, i+1] = chain[0, i] + self.struct_longueur*np.cos(angle*np.pi/180.) 
            chain[1, i+1] = chain[1, i] + self.struct_longueur*np.sin(angle*np.pi/180.) 
            structure_[2, 3+i] = +angle*np.pi/180. 
            structure_[2, i] = np.pi-angle*np.pi/180
        structure_[0, 3:] = .5*(chain[0, 1:]+chain[0, :-1])
        structure_[0, :3] = -.5*(chain[0, 1:]+chain[0, :-1])
        structure_[1, 3:] = .5*(chain[1, 1:]+chain[1, :-1])
        structure_[1, :3] = .5*(chain[1, 1:]+chain[1, :-1])
        return structure_

    def add_structure(self):
        self.N_lame += self.struct_N
        self.lames = np.hstack((self.lames, np.zeros((4, self.struct_N))))
        self.lames[:3, -self.struct_N:] = self.do_structure()
        self.lame_length = np.hstack((self.lame_length, self.struct_longueur*np.ones(self.struct_N))) # en mètres
        self.lame_width = np.hstack((self.lame_width, .042*np.ones(self.struct_N))) # en mètres
        
        
    def sample_structure(self, N_mirror=0, alpha = .8):
        struct = self.lames[:3, -self.struct_N:]
        self.particles = np.ones((3, self.N_particles))
        N_particles_ = self.N_particles/self.struct_N
        for i, vec in enumerate(struct.T.tolist()):
            x0, x1 = vec[0] - .5*self.struct_longueur*np.cos(vec[2]), vec[0] + .5*self.struct_longueur*np.cos(vec[2])
            y0, y1 = vec[1] - .5*self.struct_longueur*np.sin(vec[2]), vec[1] + .5*self.struct_longueur*np.sin(vec[2])
            self.particles[0, i*N_particles_:(i+1)*N_particles_] = np.linspace(x0, x1, N_particles_)
            self.particles[1, i*N_particles_:(i+1)*N_particles_] = np.linspace(y0, y1, N_particles_)
        
        # duplicate according to mirrors
        for i in range(N_mirror):
            particles = self.particles.copy() # the current structure to mirror
            particles_mirror = particles.copy() # the new set of particles with their mirror image
            for segment in self.structure_as_segments():
                particles_mirror = np.hstack((particles_mirror, mirror(particles, segment, alpha**(i+1))))
            # print(alpha**(i+1), particles_mirror[-1, -1])
            self.particles = particles_mirror

    def structure_as_segments(self):
        struct = self.lames[:3, -self.struct_N:]
        segments = []
        for i, vec in enumerate(struct.T.tolist()):
            x0, x1 = vec[0] - .5*self.struct_longueur*np.cos(vec[2]), vec[0] + .5*self.struct_longueur*np.cos(vec[2])
            y0, y1 = vec[1] - .5*self.struct_longueur*np.sin(vec[2]), vec[1] + .5*self.struct_longueur*np.sin(vec[2])
            segments.append(np.array([[x0, y0], [x1, y1]]).T)
        return segments
            
    def theta_E(self, im, X_, Y_, w):
        try:
            assert(self.slip.N_X==im.shape[1])
        except:
            from NeuroTools.parameters import ParameterSet
            from SLIP import Image
            from LogGabor import LogGabor
            self.slip = Image(ParameterSet({'N_X':im.shape[1], 'N_Y':im.shape[0]}))
            self.lg = LogGabor(self.slip)
        im_ = im.sum(axis=-1)
        im_ = im_ * np.exp(-.5*((.5 + .5*self.slip.x-Y_)**2+(.5 + .5*self.slip.y-X_)**2)/w**2)
        E = np.zeros((self.N_theta,))
        for i_theta, theta in enumerate(self.thetas):
            params= {'sf_0':self.sf_0, 'B_sf':self.B_sf, 'theta':theta, 'B_theta': np.pi/self.N_theta}
            FT_lg = self.lg.loggabor(0, 0, **params)
            E[i_theta] = np.sum(np.absolute(self.slip.FTfilter(np.rot90(im_, -1), FT_lg, full=True))**2)
        return E

    def theta_max(self, im, X_, Y_, w):
        E = self.theta_E(im, X_, Y_, w)
        return self.thetas[np.argmax(E)] - np.pi/2


    def theta_sobel(self, im, N_blur):
        im_ = im.copy()
        sobel = np.array([[1,   2,  1,],
                          [0,   0,  0,],
                          [-1, -2, -1,]])
        if im_.ndim==3: im_ = im_.sum(axis=-1)
        from scipy.signal import convolve2d
        im_X = convolve2d(im_, sobel, 'same')
        im_Y = convolve2d(im_, sobel.T, 'same')

        N_X, N_Y = im_.shape
        x, y = np.mgrid[0:1:1j*N_X, 0:1:1j*N_Y]
        mask = np.exp(-.5*((x-.5)**2+(y-.5)**2)/w**2)
        im_X = convolve2d(im_X, mask, 'same')
        im_Y = convolve2d(im_Y, mask, 'same')
        blur = np.array([[1, 2, 1],
                         [2, 8, 2],
                         [1, 2, 1]])
        for i in range(N_blur):
            im_X = convolve2d(im_X, blur, 'same')
            im_Y = convolve2d(im_Y, blur, 'same')

        angle = np.arctan2(im_Y, im_X)

        bord = .1
        angles = np.empty(self.N_lame)
        N_X, N_Y = im_.shape
        for i in range(self.N_lame):
            angles[i] = angle[int((bord+self.lames[0, i]*(1-2*bord))*N_X),
                              int((bord+self.lames[1, i]*(1-2*bord))*N_Y)]
        return angles - np.pi/2

    def pos_rel(self, do_torus=False):
        def torus(x, w=1.):
            """
            center x in the range [-w/2., w/2.]

            To see what this does, try out:
            >> x = np.linspace(-4,4,100)
            >> pylab.plot(x, torus(x, 2.))

            """
            return np.mod(x + w/2., w) - w/2.
        dx = self.lames[0, :, np.newaxis]-self.lames[0, np.newaxis, :]
        dy = self.lames[1, :, np.newaxis]-self.lames[1, np.newaxis, :]
        if do_torus:
            return torus(dx), torus(dy)
        else:
            return dx, dy

    def distance(self, do_torus=False):
        dx, dy = self.pos_rel(do_torus=do_torus) 
        return np.sqrt(dx **2 + dy **2)

    def angle_relatif(self):
        return self.lames[2, :, np.newaxis]-self.lames[2, np.newaxis, :]

    def angle_cocir(self, do_torus=False):
        dx, dy = self.pos_rel(do_torus=do_torus)
        theta = self.angle_relatif()
        return np.arctan2(dy, dx) - np.pi/2 - theta

    def champ(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame
        force = np.zeros_like(self.lames[2, :N_lame])
        noise = lambda t: 0.2 * np.exp((np.cos(2*np.pi*(t-0.) / 6.)-1.)/ 1.5**2)
        damp = lambda t: 0.01 #* np.exp(np.cos(t / 6.) / 3.**2)
        colin_t = lambda t: -.1*np.exp((np.cos(2*np.pi*(t-2.) / 6.)-1.)/ .3**2)
        cocir_t = lambda t: -4.*np.exp((np.cos(2*np.pi*(t-4.) / 6.)-1.)/ .5**2)
        cocir_d = lambda d: np.exp(-d/.05)
        colin_d = lambda d: np.exp(-d/.2)

        force += colin_t(self.t) * np.sum(np.sin(2*(self.angle_relatif()[:N_lame]))*colin_d(self.distance()[:N_lame]), axis=1)
        force += cocir_t(self.t) * np.sum(np.sin(2*(self.angle_cocir()[:N_lame]))*cocir_d(self.distance()[:N_lame]), axis=1)
        force += noise(self.t)*np.pi*np.random.randn(N_lame)
        force -= damp(self.t) * self.lames[3, :N_lame]/self.dt
        return 42.*force

    def update(self):
        if self.structure: N_lame = self.N_lame-self.struct_N
        else: N_lame = self.N_lame
        self.lames[2, :N_lame] += self.lames[3, :N_lame]*self.dt/2
        self.lames[3, :N_lame] += self.champ() * self.dt
        self.lames[2, :N_lame] += self.lames[3, :N_lame]*self.dt/2
        # angles are defined as non oriented between -pi/2 and pi/2 
        self.lames[2, :N_lame] = np.mod(self.lames[2, :N_lame] + np.pi/2, np.pi) - np.pi/2  

    def receive(self):
        if not self.filename is None:
            if os.path.isfile(self.filename):
                if self.structure: N_lame = self.N_lame-self.struct_N
                else: N_lame = self.N_lame
                self.t = self.time()
                i_t = np.argmin(self.z[:, 0] < np.mod(self.t, self.period))
                if self.verb: print("playback at t=", np.mod(self.t, self.period), i_t)
                self.lames[2, :N_lame] =  self.z[i_t, 1:]
                return

        if self.stream:
            if self.verb: print("Sending request")
            self.socket.send (b"Hello")
            if self.verb: print( "Received reply ", message)
            self.lames[2, :] = recv_array(self.socket)
            if self.verb: print("Received reply ", Theta.shape)
        else:
            self.dt = self.time() - self.t
            self.update()
            self.t = self.time()

#         if not self.filename is None:
#             if not os.path.isfile(self.filename):
#                 # recording
#                 if self.verb: print("recording at t=", self.t)
#                 self.z = np.vstack((self.z, np.hstack((np.array(self.t), self.lames[2, :] ))))
        return

    def render(self, fps=10, W=1000, H=618, location=[0, 1.75, -5], head_size=.4, light_intensity=1.2, reflection=1., 
               look_at=[0, 1.5, 0], fov=75, antialiasing=0.001, duration=5, fname='/tmp/temp.webm'):

        def scene(t):
            """ 
            Returns the scene at time 't' (in seconds) 
            """

            head_location = np.array(location) - np.array([0, 0, head_size])
            import vapory
            light = vapory.LightSource([15, 15, 1], 'color', [light_intensity]*3)
            background = vapory.Box([0, 0, 0], [1, 1, 1], 
                     vapory.Texture(vapory.Pigment(vapory.ImageMap('png', '"../files/VISUEL_104.png"', 'once')),
                             vapory.Finish('ambient', 1.2) ),
                     'scale', [self.background_depth, self.background_depth, 0],
                     'translate', [-self.background_depth/2, -.45*self.background_depth, -self.background_depth/2])
            me = vapory.Sphere( head_location, head_size, vapory.Texture( vapory.Pigment( 'color', [1, 0, 1] )))
            self.t = t
            self.update()
            objects = [background, me, light]

            for i_lame in range(self.N_lame):
                #print(i_lame, self.lame_length[i_lame], self.lame_width[i_lame])
                objects.append(vapory.Box([-self.lame_length[i_lame]/2, 0, -self.lame_width[i_lame]/2], 
                                          [self.lame_length[i_lame]/2, self.lames_height,  self.lame_width[i_lame]/2], 
                                           vapory.Pigment('color', [1, 1, 1]),
                                           vapory.Finish('phong', 0.8, 'reflection', reflection),
                                           'rotate', (0, -self.lames[2, i_lame]*180/np.pi, 0), #HACK?
                                           'translate', (self.lames[0, i_lame], 0, self.lames[1, i_lame])
                                          )
                              )

            objects.append(light)
            return vapory.Scene( vapory.Camera('angle', fov, "location", location, "look_at", look_at),
                           objects = objects,
                           included=["glass.inc"] )
        import moviepy.editor as mpy
        if not os.path.isfile(fname):
            self.dt = 1./fps
            def make_frame(t):
                return scene(t).render(width=W, height=H, antialiasing=antialiasing)

            clip = mpy.VideoClip(make_frame, duration=duration)
            clip.write_videofile(fname, fps=fps)
        return mpy.ipython_display(fname, fps=fps, loop=1, autoplay=1)

    def plot_structure(self, W=1000, H=618, fig=None, ax=None, border = 0.0, 
            opts = dict(vmin=-1, vmax=1., linewidths=0, cmap=None, alpha=.1, s=3.), 
            scale='auto'): #
        opts.update(cmap=plt.cm.hsv)
        if fig is None: fig = plt.figure(figsize=(self.figsize, self.figsize*H/W))
        if ax is None: ax = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        scat  = ax.scatter(self.particles[0,::-1], self.particles[1,::-1], c=self.particles[2,::-1], **opts)
        if type(scale) is float:
            ax.set_xlim([-scale, scale])
            ax.set_ylim([-scale*H/W, scale*H/W])
        elif not scale is 'auto':
            ax.set_xlim([-self.total_width, self.total_width])
            ax.set_ylim([-self.total_width*H/W, self.total_width*H/W])
        else:
            ax.set_xlim([min(self.particles[0, :].min(), self.particles[1, :].min()/H*W), 
                         max(self.particles[0, :].max(), self.particles[1, :].max()/H*W)])
            ax.set_ylim([min(self.particles[1, :].min(), self.particles[0, :].min()*H/W), 
                         max(self.particles[1, :].max(), self.particles[0, :].max()*H/W)])
        ax.axis('off') 
        return fig, ax
    
    def animate(self, fps=10, W=1000, H=618, duration=20, scale='auto', fname=None):
        if fname is None:
            import tempfile
            fname = tempfile.mktemp() + '.webm'
        import matplotlib.pyplot as plt
        self.dt = 1./fps
        inches_per_pt = 1.0/72.27
        from moviepy.video.io.bindings import mplfig_to_npimage
        import moviepy.editor as mpy
        def make_frame_mpl(t):
            self.t = t
            self.update()
            fig = plt.figure(figsize=(W*inches_per_pt, H*inches_per_pt))
            fig, ax = self.plot_structure(fig=fig, ax=None, scale=scale)
            #ax.clear()
            ax.axis('off')
            #fig, ax = self.plot_structure(fig=fig, ax=ax)
            return mplfig_to_npimage(fig) # RGB image of the figure

        animation = mpy.VideoClip(make_frame_mpl, duration=duration)
        plt.close('all')
        animation.write_videofile(fname, fps=fps)
        return mpy.ipython_display(fname, fps=fps, loop=1, autoplay=1, width=W)

    #def show_edges(self, fig=None, a=None):
        #self.N_theta = 12
        #self.thetas = np.linspace(0, np.pi, self.N_theta)
        #self.sf_0 = .3
        #self.B_sf = .3
#
        #self.vext = '.webm'
        #self.figpath = '../files/figures/elasticite/'
        #self.fps = 25
        #"""
        #Shows the quiver plot of a set of edges, optionally associated to an image.
#
        #"""
        #import pylab
        #import matplotlib.cm as cm
        #if fig==None:
            #fig = pylab.figure(figsize=(self.figsize, self.figsize))
        #if a==None:
            #border = 0.0
            #a = fig.add_axes((border, border, 1.-2*border, 1.-2*border), axisbg='w')
        #else:
            #self.update_lines()
        #marge = self.lame_length*3.
        #a.axis(self.lames_minmax + np.array([-marge, +marge, -marge, +marge]))
        #a.add_collection(self.lines)
        #a.axis(c='b', lw=0)
        #pylab.setp(a, xticks=[])
        #pylab.setp(a, yticks=[])
        #pylab.draw()
        #return fig, a

    #def set_lines(self):
        #from matplotlib.collections import LineCollection
        #import matplotlib.patches as patches
        # draw the segments
        #segments, colors, linewidths = list(), list(), list()
#
        #X, Y, Theta = self.lames[0, :], self.lames[1, :].real, self.lames[2, :]
        #for x, y, theta in zip(X, Y, Theta):
            #u_, v_ = np.cos(theta)*self.lame_length, np.sin(theta)*self.lame_length
            #segments.append([(x - u_, y - v_), (x + u_, y + v_)])
            #colors.append((0, 0, 0, 1))# black
            #linewidths.append(self.line_width)
        #return LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
#
    #def update_lines(self):
        #from matplotlib.collections import LineCollection
        #import matplotlib.patches as patches
        #X, Y, Theta = self.lames[0, :], self.lames[1, :], self.lames[2, :]
        #segments = list()
#
        #for i, (x, y, theta) in enumerate(zip(X, Y, Theta)):
            #u_, v_ = np.cos(theta)*self.lame_length, np.sin(theta)*self.lame_length
            #segments.append([(x - u_, y - v_), (x + u_, y + v_)])
        #self.lines.set_segments(segments)
#
#
    #def fname(self, name):
        #return os.path.join(self.figpath, name + self.vext)
#
    #def make_anim(self, name, make_lames, duration=3., redo=False):
        #if redo or not os.path.isfile(self.fname(name)):
#
            #import matplotlib.pyplot as plt
            #from moviepy.video.io.bindings import mplfig_to_npimage
            #import moviepy.editor as mpy
#
            #fig_mpl, ax = plt.subplots(1, figsize=(self.figsize, self.figsize), facecolor='white')
#
            #def make_frame_mpl(t):
                # on ne peut changer que l'orientation des lames:
                #self.t = t
                #self.lames[2, :] = make_lames(self)
                #self.update_lines()
                #fig_mpl, ax = self.show_edges()#fig_mpl, ax)
                #self.t_old = t
                #return mplfig_to_npimage(fig_mpl) # RGB image of the figure
#
            #animation = mpy.VideoClip(make_frame_mpl, duration=duration)
            #animation.write_videofile(self.fname(name), fps=self.fps)
#
    #def ipython_display(self, name, loop=True, autoplay=True, controls=True):
        #"""
        #showing the grid in the notebook by pointing at the file stored in the proper folder
#
        #"""
        #import os
        #from IPython.core.display import display, Image, HTML
        #opts = ' '
        #if loop: opts += 'loop="1" '
        #if autoplay: opts += 'autoplay="1" '
        #if controls: opts += 'controls '
        #s = """
            #<center><table border=none width=100% height=100%>
            #<tr> <td width=100%><center><video {0} src="{2}" type="video/{1}"  width=100%>
            #</td></tr></table></center>""".format(opts, self.vext[1:], self.fname(name))
        #return display(HTML(s))
#
try:
    import pyglet
    from pyglet.gl.glu import gluLookAt
    import pyglet.gl as gl
    smoothConfig = gl.Config(sample_buffers=1, samples=4,
                             depth_size=16, double_buffer=True)

    class Window(pyglet.window.Window):
        """
        Viewing particles using pyglet.app

            Interaction keyboard:
            - TAB pour passer/sortir du fulscreen
            - espace : passage en first-person perspective

            Les interactions visuo - sonores sont simulées ici par des switches lançant des phases:
            - F : faster
            - S : slower

        """
        def __init__(self, e, *args, **kwargs):
            #super(Window, self).__init__(*args, **kwargs)
            super(Window, self).__init__(config=smoothConfig, *args, **kwargs)
            self.e = e
            self.first_frame = True

        #@self.event
        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.TAB:
                if self.fullscreen:
                    self.set_fullscreen(False)
                    self.set_location(screen.width/3, screen.height/3)
                else:
                    self.set_fullscreen(True)
            elif symbol == pyglet.window.key.ESCAPE:
                pyglet.app.exit()
    #
        #@self.win.event
        def on_resize(self, width, height):
            print('The window was resized to %dx%d' % (width, height))
    #
        #@self.win.event
        def on_draw(self):
            if (not self.e.period is None) and (not self.e.filename is None):
                if True: #not os.path.isfile(self.e.filename):
                    if self.e.time() > self.e.period:
                        pyglet.app.exit()
            self.clear()
            gl.glMatrixMode(gl.GL_PROJECTION);
            gl.glLoadIdentity()
    #                     gluOrtho2D sets up a two-dimensional orthographic viewing region.  
    #          Parameters left, right
    #                             Specify the coordinates for the left and right vertical clipping planes.
    #                         bottom, top
    #                             Specify the coordinates for the bottom and top horizontal clipping planes.
    #                         Description
    #         gl.gluOrtho2D(-(self.W-1)/2*self.e.total_width, (self.W+1)/2*self.e.total_width, -self.e.total_width/2, self.e.total_width/2, 0, 0, 1);
            self.W = float(self.width)/self.height
            gl.gluOrtho2D(-self.W/2*self.e.total_width, self.W/2*self.e.total_width, -self.e.total_width/2, self.e.total_width/2, 0, 0, 1);
            gl.glMatrixMode(gl.GL_MODELVIEW);
            gl.glLoadIdentity();

            #gl.glLineWidth () #p['line_width'])
            gl.glEnable (gl.GL_BLEND)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glColor3f(0., 0., 0.)


            if self.first_frame:
#                 print(self.e.t0, self.e.t)
                self.e.t0 = time.time()
                self.e.t = self.e.time()
#                 print(self.e.t0, self.e.t)
                self.first_frame = False

            dt = self.e.time() - self.e.t
            if 1./self.e.desired_fps - dt>0.: time.sleep(1./self.e.desired_fps - dt)

            self.e.receive()
            X, Y, Theta = self.e.lames[0, :], self.e.lames[1, :], self.e.lames[2, :]
            dX, dY = np.cos(Theta)/2., np.sin(Theta)/2.
            #print(Theta)
            # coords = np.vstack((X-dX*self.e.lame_length, Y-dY*self.e.lame_length, X+dX*self.e.lame_length, Y+dY*self.e.lame_length))
            coords = np.vstack((
                                X-dX*self.e.lame_length+dY*self.e.lame_width, Y-dY*self.e.lame_length-dX*self.e.lame_width,
                                X+dX*self.e.lame_length+dY*self.e.lame_width, Y+dY*self.e.lame_length-dX*self.e.lame_width,
                                X-dX*self.e.lame_length-dY*self.e.lame_width, Y-dY*self.e.lame_length+dX*self.e.lame_width,
                                X+dX*self.e.lame_length-dY*self.e.lame_width, Y+dY*self.e.lame_length+dX*self.e.lame_width,
                                ))
            #pyglet.graphics.draw(2*self.e.N_lame, gl.GL_LINES, ('v2f', coords.T.ravel().tolist()))
            indices = np.array([0, 1, 2, 1, 2, 3])[:, np.newaxis] + 4*np.arange(self.e.N_lame)
            pyglet.graphics.draw_indexed(4*self.e.N_lame, pyglet.gl.GL_TRIANGLES,
                                         indices.T.ravel().tolist(),
                                         ('v2f', coords.T.ravel().tolist()))
            #pyglet.graphics.draw(4*self.e.N_lame, gl.GL_QUADS, ('v2f', coords.T.ravel().tolist()))
            # carré
            if self.e.DEBUG:
                coords = np.array([[-.5*self.e.total_width, .5*self.e.total_width, .5*self.e.total_width, -.5*self.e.total_width], [-.5*self.e.total_width, -.5*self.e.total_width, .5*self.e.total_width, .5*self.e.total_width]])
                pyglet.graphics.draw(4, gl.GL_LINE_LOOP, ('v2f', coords.T.ravel().tolist()))
            # centres des lames
            if self.e.DEBUG:
                gl.glLineWidth (1.)
                gl.glColor3f(0., 0., 0.)
                pyglet.graphics.draw(self.e.N_lame, gl.GL_POINTS, ('v2f', self.e.lames[:2,:].T.ravel().tolist()))
                gl.glColor3f(1., 0., 0.)
                pyglet.graphics.draw(2, gl.GL_LINES, ('v2f', [0., 0., 1., 0.]))
                gl.glColor3f(0., 1., 0.)
                pyglet.graphics.draw(2, gl.GL_LINES, ('v2f', [0., 0., 0., 1.]))
except:
    print('Could not load pyglet')

def server(e):
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % e.port)
    if e.verb: print("Running server on port: ", e.port)
    # serves only 5 request and dies
    while True:
        # Wait for next request from client
        message = socket.recv()
        if e.verb: print("Received request %s" % message)
        e.dt = e.time() - e.t
        #e.update()
        e.receive()
        e.t = e.time()
        send_array(socket, e.lames[2, :])

def serial(e):
    e.load()
    import serial
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    def message(i, increment):
#         return alphabet[i] + str(increment)  + ';\n'
        return alphabet[i] + str(increment)  + ';'

    def convert(increment):
        msg  = ''
        for i, increment_ in enumerate(increment):
            msg += message(i, increment)
        return msg

    with serial.Serial(e.serial_port, e.baud_rate) as ser:
        if e.structure: N_lame = e.N_lame-e.struct_N
        else: N_lame = e.N_lame
        if e.verb: print("Running serial on port: ", e.serial_port)
        angle_actuel = np.zeros_like(e.lames[2, :N_lame])
        while True:
            #e.dt = e.time() - e.t
            #e.update()
            #e.t = e.time()
            e.receive()
            #angle_desire = [int(theta/2/np.pi*e.n_pas) for theta in e.lames[2, :N_lame]]
            angle_desire = e.lames[2, :N_lame]
            # on transforme en l'angle à faire pour obtenir la bonne position
            d_angle = np.mod((angle_desire - angle_actuel) + np.pi/2, np.pi) - np.pi/2
            # et donc du nombre de pas à faire
            dnbpas =  d_angle/2/np.pi*e.n_pas
            # HACK : écrétage pour éviter un overflow
            dnbpas = e.n_pas_max * np.tanh(dnbpas/e.n_pas_max)
            # on convertit en int
            dnbpas = dnbpas.astype(np.int)
            print(e.lames[2, :N_lame], angle_desire, angle_actuel, dnbpas)
            angle_actuel = angle_actuel + dnbpas*2*np.pi/e.n_pas
            angle_actuel = np.mod(angle_actuel + np.pi/2, np.pi) - np.pi/2
            # if e.verb: print('@', e.t, convert(dnbpas), '-fps=', 1./e.dt)
            if e.verb: print('@', e.t, '-fps=', 1./e.dt)
            #ser.write(convert(dnbpas))
            for i, increment in enumerate(dnbpas):
                if np.abs(increment) > e.n_pas_max:
                    print('!! /Z\ !! @ ', e.t, ' overflow @ ', i, increment)
                msg = message(i, increment)
                ser.write(msg)
                # if e.verb and i==0: print (msg)
            dt = e.time() - e.t
            if 1./e.desired_fps - dt>0.: time.sleep(1./e.desired_fps - dt)


def writer(e, force=False):
    if not e.filename is None:
        if not os.path.isfile(e.filename) or force:
            if e.structure: N_lame = e.N_lame-e.struct_N
            else: N_lame = e.N_lame
            e.z = np.zeros((0, N_lame+1))
            for t in np.arange(0., e.period, 1./e.desired_fps):
                e.dt = 1./e.desired_fps
                e.t = t
                e.update()
                if e.verb: print("recording at t=", e.t)
                e.z = np.vstack((e.z, np.hstack((np.array(e.t), e.lames[2, :N_lame] ))))
            # save the file
            np.save(e.filename, e.z)

def client(e):
    writer(e)
    e.load()
    if e.stream:
        import zmq
        context = zmq.Context()
        if e.verb: print("Connecting to server with port %s" % e.port)
        e.socket = context.socket(zmq.REQ)
        e.socket.connect ("tcp://localhost:%s" % e.port)

    platform = pyglet.window.get_platform()
    print("platform" , platform)
    display = platform.get_default_display()
    print("display" , display)
    screens = display.get_screens()
    print("screens" , screens)
    for i, screen in enumerate(screens):
        print('Screen %d: %dx%d at (%d,%d)' % (i, screen.width, screen.height, screen.x, screen.y))
    N_screen = len(screens) # number of screens
    N_screen = 1# len(screens) # number of screens
    assert N_screen == 1 # we should be running on one screen only
    def callback(dt):
        if e.verb: print('%f seconds since last callback' % dt , '%f  fps' % pyglet.clock.get_fps())
        pass
    window = Window(e, width=screen.width*2/3, height=screen.height*2/3)
    window.set_location(screen.width/3, screen.height/3)
    pyglet.gl.glClearColor(1., 1., 1., 1.)
    pyglet.clock.schedule(callback)
#     window.e.t0 = window.e.time(init=True)
    pyglet.app.run()
    if e.screenshot: pyglet.image.get_buffer_manager().get_color_buffer().save('screenshot.png')

def main(e):
    if e.display:
        # Now we can connect a client to the server
        client(e)

    elif e.stream:
        # Now we can run the server
        server(e)

    elif e.serial:
        serial(e)

    else:
        writer(e, force=True)

if __name__ == '__main__':
    e = EdgeGrid()
    main(e)
