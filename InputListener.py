from pynput import mouse, keyboard
import numpy as np
from pynput.keyboard import Key
from pynput.mouse import Button


class InputListener:
    keyboardListener = None
    mouseListener = None
    mouseController = None

    leftPress = False
    lastpos = None
    currpos = None

    forward = False
    backward = False
    left = False
    right = False
    up = False
    down = False
    pause = False
    toggle_pause = False
    esc = False

    def __init__(self):
        self.keyboardListener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.mouseListener = mouse.Listener(on_click=self.on_click)



        self.mouseController = mouse.Controller()

        self.lastpos = self.getPos()

    def on_key_press(self, key):
        try:
            if isinstance(key, keyboard.KeyCode):
                if key.char == 'w' and not self.forward:
                    self.forward = True
                if key.char == 's' and not self.backward:
                    self.backward = True
                if key.char == 'a' and not self.left:
                    self.left = True
                if key.char == 'd' and not self.right:
                    self.right = True
                if key.char == 'p' and not self.toggle_pause:
                    self.toggle_pause = True

            if isinstance(key, keyboard.Key):
                if key == Key.shift and not self.down:
                    self.down = True
                if key == Key.space and not self.up:
                    self.up = True
                if key == Key.esc and not self.esc:
                    self.esc = True

        except AttributeError:
            #print(f"Special key pressed: {key}")  # For special keys like shift, ctrl, etc.
            a = 0

        return True

    def on_key_release(self, key):
        try:
            if isinstance(key, keyboard.KeyCode):
                if key.char == 'w' and self.forward:
                    self.forward = False
                if key.char == 's' and self.backward:
                    self.backward = False
                if key.char == 'a' and self.left:
                    self.left = False
                if key.char == 'd' and self.right:
                    self.right = False
                if key.char == 'p' and self.toggle_pause:
                    self.pause = not self.pause
                    self.toggle_pause = False

            if isinstance(key, keyboard.Key):
                if key == Key.shift and self.down:
                    self.down = False
                if key == Key.space and self.up:
                    self.up = False
                if key == Key.esc and self.esc:
                    self.esc = False


        except AttributeError:
            a = 0

        return True

    # Mouse event handling methods
    def on_click(self, x, y, button, pressed):
        if pressed and Button.left:
            self.lastpos = self.getPos()
            self.leftPress = True
        else:
            self.lastpos = self.getPos()
            self.leftPress = False

        return True

    def getPos(self):
        return np.array([self.mouseController.position[0], self.mouseController.position[1]])

    def getMouseDelta(self):
        currpos = self.getPos()
        delta = currpos - self.lastpos
        self.lastpos = currpos
        return delta.astype(np.float64)

    def getLeft(self):
        return self.leftPress

    def listen(self):
        # Starting the keyboard and mouse listeners in separate threads
        self.keyboardListener.start()
        self.mouseListener.start()

    def __del__(self):
        # Join both threads so the program waits for both listeners to finish
        self.keyboardListener.join()
        self.mouseListener.join()

    def getPause(self):
        return self.pause