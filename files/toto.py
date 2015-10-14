import pyglet

def main():
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

    #if True:# do_fs:
        #self.win = Window(screen=screens[0], fullscreen=True, resizable=True)
    #else:
        #self.win = Window(width=screen.width*2/3, height=screen.height*2/3, screen=screens[0], fullscreen=False, resizable=True)
        #self.win.set_location(screen.width/3, screen.height/3)
#
    #self.fps_text = pyglet.clock.ClockDisplay()

    window = pyglet.window.Window(width=960, height=700)
    pyglet.app.run()

if __name__ == '__main__':
    main()


