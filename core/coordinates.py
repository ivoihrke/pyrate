# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 12:36:11 2016

Pyrate - Optical raytracing based on Python

Copyright (C) 2014 Moritz Esslinger moritz.esslinger@web.de
               and Johannes Hartung j.hartung@gmx.net
               and    Uwe Lippmann  uwe.lippmann@web.de

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

@author: Johannes Hartung
"""

import numpy as np
import math
import uuid

from optimize import ClassWithOptimizableVariables, OptimizableVariable

#TODO: want to have some aiming function aimAt(ref), aimAt(globalcoords)?


# TODO: to avoid circle references: please add addChild function
# TODO: how to arange names that dict has unique identifier and name is still
# changeable? mydict[new_key] = mydict.pop(old_key)

class LocalCoordinates(ClassWithOptimizableVariables):
    def __init__(self, name="", **kwargs):
        '''
        @param: name -- string for identification
        @param: kwargs -- keyword args: decz, decx, decy, tiltx, tilty, tiltz, order        
        
        '''
        super(LocalCoordinates, self).__init__()        

        #thickness=0, decx=0, decy=0, tiltx=0, tilty=0, tiltz=0, order=0        
        
        (decz, decx, decy, tiltx, tilty, tiltz, order) = \
        (kwargs.get(key, 0.0) for key in ["decz", "decx", "decy", "tiltx", "tilty", "tiltz", "order"])
        
        
        if name == "":
            name = str(uuid.uuid1())
        
        self.setName(name)
        
        

        self.decx = OptimizableVariable(variable_status=False, variable_type='Variable', value=decx)
        self.decy = OptimizableVariable(variable_status=False, variable_type='Variable', value=decy)
        self.decz = OptimizableVariable(variable_status=False, variable_type='Variable', value=decz)
        self.tiltx = OptimizableVariable(variable_status=False, variable_type='Variable', value=tiltx)
        self.tilty = OptimizableVariable(variable_status=False, variable_type='Variable', value=tilty)
        self.tiltz = OptimizableVariable(variable_status=False, variable_type='Variable', value=tiltz)
        


        self.addVariable("decx", self.decx)
        self.addVariable("decy", self.decy)
        self.addVariable("decz", self.decz)
        self.addVariable("tiltx", self.tiltx)
        self.addVariable("tilty", self.tilty)
        self.addVariable("tiltz", self.tiltz)
        
        
        self.order = order
        
        self.parent = None # None means reference to global coordinate system 
        self.__children = [] # children
    
        self.globalcoordinates = np.array([0, 0, 0])
        self.localdecenter = np.array([0, 0, 0])
        self.localrotation = np.lib.eye(3)
        self.localbasis = np.lib.eye(3)

        self.update() # initial update

    def setName(self, name):
        if name == "":
            name = str(uuid.uuid1())
        self.__name = name
        
    def getName(self):
        return self.__name
        
    name = property(getName, setName)


    def addChild(self, tmplc):        
        tmplc.parent = self
        tmplc.update()
        self.__children.append(tmplc)
        return tmplc
        
    def addChildToReference(self, refname, tmplc):
        if self.name == refname:
            self.addChild(tmplc)
        else:
            for x in self.__children:
                x.addChildToReference(refname, tmplc)
        return tmplc
    

    def rodrigues(self, angle, a):
        ''' 
        returns numpy matrix from Rodrigues formula.
        
        @param: (float) angle in radians
        @param: (numpy (3x1)) axis of rotation (unit vector)
        
        @return: (numpy (3x3)) matrix of rotation
        '''
        mat = np.array(\
            [[    0, -a[2],  a[1]],\
             [ a[2],     0, -a[0]],\
             [-a[1],  a[0],    0]]\
             )
        return np.lib.eye(3) + math.sin(angle)*mat + (1. - math.cos(angle))*np.dot(mat, mat)
    
    def calculate(self):
        
        # order=0: decx, decy, tiltx, tilty, tiltz
        # order=1: tiltx, tilty, tilty, decx, decy        
        
        # 0 objectdist angle0
        # 1 thickness angle1
        # 2 thickness angle2       
        
        # notice negative signs for angles to make sure that for tiltx>0 the
        # optical points in positive y-direction although the x-axis of the
        # local coordinate system points INTO the screen
        # This leads also to a clocking in the mathematical negative direction
        
        tiltx = self.tiltx.evaluate()
        tilty = self.tilty.evaluate()
        tiltz = self.tiltz.evaluate()
        decx = self.decx.evaluate()
        decy = self.decy.evaluate()        
        decz = self.decz.evaluate()
        
        self.localdecenter = np.array([decx, decy, decz])
        if self.order == 0:
            self.localrotation = np.dot(self.rodrigues(tiltz, [0, 0, 1]), np.dot(self.rodrigues(tilty, [0, 1, 0]), self.rodrigues(tiltx, [1, 0, 0])))
        else:
            self.localrotation = np.dot(self.rodrigues(tiltx, [1, 0, 0]), np.dot(self.rodrigues(tilty, [0, 1, 0]), self.rodrigues(tiltz, [0, 0, 1])))
            

    def update(self):
        ''' 
        runs through all references specified and sums up 
        coordinates and local rotations to get appropriate 
        global coordinate
        '''
        self.calculate()

        parentcoordinates = np.array([0, 0, 0])
        parentbasis = np.lib.eye(3)

        if self.parent != None:        
            parentcoordinates = self.parent.globalcoordinates
            parentbasis = self.parent.localbasis
        
        self.localbasis = np.dot(self.localrotation, parentbasis)
        if self.order == 0:
            # first decenter then rotation, afterwards thickness
            self.globalcoordinates = \
            parentcoordinates + \
            np.dot(parentbasis, self.localdecenter)
            # TODO: removed .T on parentbasis to obtain correct behavior; examine!
        else:
            # first rotation then decenter, afterwards thickness
            self.globalcoordinates = \
            parentcoordinates + \
            np.dot(self.localbasis, self.localdecenter)
            # TODO: removed .T on localbasis to obtain correct behavior; examine!
            
        for ch in self.__children:
            ch.update()

    def returnLocalToGlobalPoints(self, localpts):
        """
        @param: localpts (3xN numpy array)
        @return: globalpts (3xN numpy array)
        """
        transformedlocalpts = np.dot(self.localbasis, localpts)
        # construction to use broadcasting        
        globalpts = (transformedlocalpts.T + self.globalcoordinates).T
        return globalpts

    def returnLocalToGlobalDirections(self, localdirs):
        """
        @param: localdirs (3xN numpy array)
        @return: globaldirs (3xN numpy array)
        """
        return np.dot(self.localbasis, localdirs)

    def returnGlobalToLocalPoints(self, globalpts):
        """
        @param: globalpts (3xN numpy array)
        @return: localpts (3xN numpy array)
        """
        translatedglobalpts = (globalpts.T - self.globalcoordinates).T
        # construction to use broadcasting
        localpts = np.dot(self.localbasis.T, translatedglobalpts)        
        return localpts

    def returnGlobalToLocalDirections(self, globaldirs):
        """
        @param: globaldirs (3xN numpy array)
        @return: localdirs (3xN numpy array)
        """
        
        localpts = np.dot(self.localbasis.T, globaldirs)        
        return localpts
        

    def returnConnectedNames(self):
        lst = [self.name]
        for ch in self.__children:
            lst = lst + ch.returnConnectedNames()
        return lst
        
    def pprint(self, n=0):
        s = n*"    " + self.name + " (" + str(self.globalcoordinates) + ")\n"
        for x in self.__children:
            s += x.pprint(n+1)
            
        return s
        

    def __str__(self):
        s = 'name %s\norder %d\nglobal coordinates: %s\nld: %s\nlr: %s\nlb: %s\nchildren %s'\
        % (self.name, self.order, \
        self.globalcoordinates, \
        self.localdecenter, \
        self.localrotation, \
        self.localbasis, \
        [i.name for i in self.__children])
        return s
    
    
origin = LocalCoordinates()

if __name__ == "__main__":
    '''testcase1: undo coordinate break'''
    surfcb1 = LocalCoordinates(name="1", decz=40.0)
    surfcb2 = surfcb1.addChild(LocalCoordinates(name="2", decz=20.0))
    surfcb3 = surfcb2.addChild(LocalCoordinates(name="3", decy=15.0, tiltx=10.0*math.pi/180.0, order=0))
    surfcb35 = surfcb3.addChild(LocalCoordinates(name="35", decz=-20.0))
    surfcb4 = surfcb35.addChild(LocalCoordinates(name="4"))
    surfcb45 = surfcb4.addChild(LocalCoordinates(name="45", decz=+20.0))
    surfcb5 = surfcb45.addChild(LocalCoordinates(name="5", decy=-15.0, tiltx=-10.0*math.pi/180.0, order=1))
    surfcb6 = surfcb5.addChild(LocalCoordinates(name="6", decz=-20.0))
    surfcb7 = surfcb6.addChild(LocalCoordinates(name="7"))
    
    surfcb8 = surfcb3.addChild(LocalCoordinates(name="8", decx=5.555, decy=3.333))    
    surfcb9 = surfcb3.addChild(LocalCoordinates(name="9", decx=-5.555, decy=-3.333))    
    
    print(str(surfcb1))
    print(str(surfcb2))
    print(str(surfcb3))
    print(str(surfcb35))
    print(str(surfcb4))
    print(str(surfcb45))
    print(str(surfcb5))
    print(str(surfcb6))
    print(str(surfcb7))
    
    print(surfcb1.returnConnectedNames())
    print(surfcb1.pprint())

    o = np.random.random((3,5))
    print("ORIGINAL")
    print(o)
    print("TRANSFORMED")    
    print(surfcb4.returnGlobalToLocalPoints(surfcb4.returnLocalToGlobalPoints(o)))