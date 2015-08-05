# standard include

import time
import math

import numpy as np
import matplotlib.pyplot as plt
from PySide import QtCore, QtGui

import core.material
import core.surfShape
import core.aim
import core.field
import core.pupil
import core.raster
import core.plots

from core.ray import RayPath
from core.optical_system import OpticalSystem, Surface

# freecad modules

import FreeCAD
import Part
import Points

class AimDialog(QtGui.QDialog):
    def __init__(self, pupilsize, stopposition):
        super(AimDialog, self).__init__()

        self.pupiltype = ""
        self.fieldtype = ""
        self.rastertype = ""
        self.pupilsize = pupilsize
        self.stopposition = stopposition

        self.initUI()

    def initUI(self):

        #pupiltype = core.pupil.EntrancePupilDiameter
        #pupilsize = 2.0
        #fieldType = core.field.ObjectHeight
        #rasterType = core.raster.RectGrid
        #stopPosition = 1


        lblpt = QtGui.QLabel("Pupil type", self)
        lblpt.move(10, 10)

        self.combopt = QtGui.QComboBox(self)
        self.combopt.addItem("EntrancePupilDiameter")
        self.combopt.addItem("EntrancePupilRadius")
        self.combopt.addItem("StopDiameter")
        self.combopt.addItem("StopRadius")
        self.combopt.addItem("ExitPupilDiameter")
        self.combopt.addItem("ExitPupilRadius")
        self.combopt.addItem("InfiniteConjugateImageSpaceFNumber")
        self.combopt.addItem("InfiniteConjugateObjectSpaceFNumber")
        self.combopt.addItem("WorkingImageSpaceFNumber")
        self.combopt.addItem("WorkingObjectSpaceFNumber")
        self.combopt.addItem("ObjectSpaceNA")
        self.combopt.addItem("ImageSpaceNA")
        self.combopt.move(10, 50)
        self.combopt.activated[str].connect(self.onActivatedPT)

        lblft = QtGui.QLabel("Field type", self)
        lblft.move(10, 90)
        self.comboft = QtGui.QComboBox(self)
        self.comboft.addItem("ObjectHeight")
        self.comboft.addItem("ObjectChiefAngle")
        self.comboft.addItem("ParaxialImageHeight")
        self.comboft.move(10, 130)
        self.comboft.activated[str].connect(self.onActivatedFT)


        lblrt = QtGui.QLabel("Raster type", self)
        lblrt.move(10, 170)


        self.combort = QtGui.QComboBox(self)
        self.combort.addItem("RectGrid")
        self.combort.addItem("HexGrid")
        self.combort.addItem("RandomGrid")
        self.combort.addItem("PoissonDiskSampling")
        self.combort.addItem("MeridionalFan")
        self.combort.addItem("SagitalFan")
        self.combort.addItem("ChiefAndComa")
        self.combort.addItem("Single")

        self.combort.move(10, 210)
        self.combort.activated[str].connect(self.onActivatedRT)

        lblps = QtGui.QLabel("Pupil size [mm]", self)
        lblps.move(410, 10)


        self.spinboxps = QtGui.QDoubleSpinBox(self)
        self.spinboxps.setValue(self.pupilsize)
        self.spinboxps.setMinimum(0.0)
        self.spinboxps.setMaximum(100.0)

        self.spinboxps.move(410, 50)
        self.spinboxps.valueChanged.connect(self.onChangedPS)


        lblst = QtGui.QLabel("Stop position (surface no.)", self)
        lblst.move(410, 90)


        self.spinboxst = QtGui.QSpinBox(self)
        self.spinboxst.setValue(self.stopposition)
        self.spinboxst.setMinimum(0)
        self.spinboxst.setMaximum(20)

        self.spinboxst.move(410, 130)

        okbtn = QtGui.QPushButton("OK",self)
        okbtn.setAutoDefault(True)
        okbtn.move(300, 250)

        okbtn.clicked.connect(self.onOK)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)

        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)

        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Aim Configuration Dialog')

        self.show()

    def onOK(self):
        self.pupiltype = self.combopt.currentText()
        self.pupilsize = self.spinboxps.value()
        self.rastertype = self.combort.currentText()
        self.fieldtype = self.comboft.currentText()
        self.stopposition = self.spinboxst.value()
        self.close()


    def onActivatedPT(self, text):
        #FreeCAD.Console.PrintMessage(text)
        self.pupiltype = text

    def onActivatedFT(self, text):
        #FreeCAD.Console.PrintMessage(text)
        self.fieldtype = text

    def onActivatedRT(self, text):
        #FreeCAD.Console.PrintMessage(text)
        self.rastertype = text


    def onChangedPS(self, val):
        #FreeCAD.Console.PrintMessage(str(val))
        self.pupilsize = val

    def onChangedST(self, val):
        #FreeCAD.Console.PrintMessage(str(val))
        self.stopposition = val


        #self.lbl.setText(text)
        #self.lbl.adjustSize()

class FieldDialog(QtGui.QDialog):
    def __init__(self, fieldpts, wavelength):
        super(FieldDialog, self).__init__()

        self.fieldpoints = fieldpts
        self.wavelength = wavelength

        self.initUI()

    def initUI(self):

        lbltable = QtGui.QLabel('Field points', self)
        lbltable.move(10,10)

        self.tableWidget = QtGui.QTableWidget(self)
        self.tableWidget.setRowCount(len(self.fieldpoints))
        self.tableWidget.setColumnCount(2)
        self.tableWidget.move(10,50)
        self.tableWidget.setHorizontalHeaderLabels(['fx', 'fy'])

        for index, fp in enumerate(self.fieldpoints):
            self.tableWidget.setItem(index, 0, QtGui.QTableWidgetItem(str(fp[0])))
            self.tableWidget.setItem(index, 1, QtGui.QTableWidgetItem(str(fp[1])))

        lblwavelength = QtGui.QLabel('Wavelength [um]', self)
        lblwavelength.move(10, 310)

        self.spinboxwl = QtGui.QDoubleSpinBox(self)
        self.spinboxwl.setValue(self.wavelength)
        self.spinboxwl.move(10, 350)

        okbtn = QtGui.QPushButton('OK', self)
        okbtn.move(10, 390)
        okbtn.clicked.connect(self.onOK)

        appendbtn = QtGui.QPushButton('Append', self)
        appendbtn.move(110, 390)
        appendbtn.clicked.connect(self.onAppend)

        #fillbtn = QtGui.QPushButton('Fill', self)
        #fillbtn.move(210, 390)
        #fillbtn.clicked.connect(self.onFill)



        self.setGeometry(300, 300, 600, 500)
        self.setWindowTitle('Field Configuration Dialog')
        self.show()

    def onFill(self):
        for r in range(self.tableWidget.rowCount()):
            for c in range(self.tableWidget.columnCount()):
                tableitem = self.tableWidget.item(r, c)
                if tableitem == None:
                    #newstr = str(-1.0 + 2.0*np.random.random())
                    #FreeCAD.Console.PrintMessage(newstr)
                    self.tableWidget.setItem(r, c, QtGui.QTableWidgetItem(str(0.0)))

    def onAppend(self):
        self.tableWidget.insertRow(self.tableWidget.rowCount())


    def onOK(self):

        self.fieldpoints[:] = []
        for rowindex in range(self.tableWidget.rowCount()):
            self.fieldpoints.append( \
                [float(self.tableWidget.item(rowindex, colindex).text()) \
                 for colindex in range(self.tableWidget.columnCount())
                ]
            )

        self.wavelength = self.spinboxwl.value()



        self.close()





class OpticalSystemInterface(object):

    def __init__(self):
        self.surfaceviews = []
        self.surfaceobs = []
        self.rayobs = []
        self.rayviews = []
        self.intersectptsobs = []

        self.aiminitialized = False
        self.fieldwaveinitialized = False
        self.fieldpoints = []
        self.wavelength = 0.55
        self.aimfinitestopdata = None

        self.os = OpticalSystem()

    def dummycreate(self): # should only create the demo system, will be removed later
        self.os = OpticalSystem() # reinit os
        self.os.surfaces[0].thickness.val = 2.0 # it is not good give the object itself a thickness if the user is not aware of that
        self.os.surfaces[1].shape.sdia.val = 1e10 # radius of image plane may not be zero to be sure to catch all rays
        self.os.insertSurface(1, Surface(core.surfShape.Conic(curv=1/-5.922, semidiam=0.55), thickness=3.0, material=core.material.ConstantIndexGlass(1.7))) # 0.55
        self.os.insertSurface(2, Surface(core.surfShape.Conic(curv=1/-3.160, semidiam=1.0), thickness=5.0)) # 1.0
        self.os.insertSurface(3, Surface(core.surfShape.Conic(curv=1/15.884, semidiam=1.3), thickness=3.0, material=core.material.ConstantIndexGlass(1.7))) # 1.3
        self.os.insertSurface(4, Surface(core.surfShape.Conic(curv=1/-12.756, semidiam=1.3), thickness=3.0)) # 1.3
        self.os.insertSurface(5, Surface(core.surfShape.Conic(semidiam=1.01), thickness=2.0)) # semidiam=1.01 # STOP
        self.os.insertSurface(6, Surface(core.surfShape.Conic(curv=1/3.125, semidiam=1.0), thickness=3.0, material=core.material.ConstantIndexGlass(1.5))) # semidiam=1.0
        self.os.insertSurface(7, Surface(core.surfShape.Conic(curv=1/1.479, semidiam=1.0), thickness=19.0)) # semidiam=1.0

    def dummycreate2(self):
        self.os = OpticalSystem() # reinit os
        self.os.surfaces[0].thickness.val = 20.0
        self.os.surfaces[1].shape.sdia.val = 1e10
        self.os.insertSurface(1, Surface(core.surfShape.Conic(curv=-1./24.,semidiam=5.0), thickness = -30.0, material=core.material.Mirror()))


    def makeSurfaceFromSag(self, surface, startpoint = [0,0,0], R=50.0, rpoints=10, phipoints=12):
        surPoints = []
        pts = Points.Points()
        pclpoints = []
        for r in np.linspace(0,R,rpoints):
            points = []
            for a in np.linspace(0.0, 360.0-360/float(phipoints), phipoints):
                x = r * math.cos(a*math.pi/180.0)# + startpoint[0]
                y = r * math.sin(a*math.pi/180.0)# + startpoint[1]
                z = surface.shape.getSag(x, y)# + startpoint[2]
                p = FreeCAD.Base.Vector(x,y,z)
                p2 = FreeCAD.Base.Vector(x+startpoint[0], y+startpoint[1], z+startpoint[2])
                points.append(p)
                pclpoints.append(p2)
            surPoints.append(points)
        pts.addPoints(pclpoints)
        sur = Part.BSplineSurface()
        sur.interpolate(surPoints)
        sur.setVPeriodic()
        surshape  = sur.toShape()
        surshape.translate(tuple(startpoint))
        return (surshape, pts)


    def createSurfaceViews(self):
        offset = [0, 0, 0]
        doc = FreeCAD.ActiveDocument

        for (index, surf) in enumerate(self.os.surfaces):
            # all accesses to the surface internal variables should be performed by appropriate supervised functions
            # to not violate the privacy of the class

            FCsurfaceobj = doc.addObject("Part::Feature", "Surf_"+str(index))
            FCsurfaceview = FCsurfaceobj.ViewObject

            if surf.shape.sdia.val > 0.5 and surf.shape.sdia.val < 100.0: # boundaries for drawing the surfaces, should be substituted by appropriate drawing conditions
                (FCsurface, FCptcloud) = self.makeSurfaceFromSag(surf, offset, surf.shape.sdia.val, 10, 36)
                #Points.show(ptcloud)
                #Part.show(surface)

                #pointcloudview = ptcloud.ViewObject
                FCsurfaceobj.Shape = FCsurface

                FCsurfaceview.ShapeColor = (0.0, 0.0, 1.0)
                #surfaceview.show()
            else:
                FCplaneshape = Part.makePlane(2,2)
                newoffset = [offset[0] - 1, offset[1] - 1, offset[2]]
                FCplaneshape.translate(tuple(newoffset))
                FCsurfaceobj.Shape = FCplaneshape

                FCsurfaceview.ShapeColor = (1.0, 0.0, 0.0)

            offset[2] += surf.getThickness() # may be substituted later by a real coordinate transformation (coordinate break)

            #time.sleep(0.1)

            self.surfaceobs.append(FCsurfaceobj) # update lists
            self.surfaceviews.append(FCsurfaceview) # update lists

        doc.recompute()


    def makeRayBundle(self, raybundle, offset):
        raysorigin = raybundle.o
        nrays = np.shape(raysorigin)[1]

        pp = Points.Points()
        sectionpoints = []

        res = []

        for i in range(nrays):
            if abs(raybundle.t[i]) > 1e-6:
                x1 = raysorigin[0, i] + offset[0]
                y1 = raysorigin[1, i] + offset[1]
                z1 = raysorigin[2, i] + offset[2]

                x2 = x1 + raybundle.t[i] * raybundle.rayDir[0, i]
                y2 = y1 + raybundle.t[i] * raybundle.rayDir[1, i]
                z2 = z1 + raybundle.t[i] * raybundle.rayDir[2, i]

                res.append(Part.makeLine((x1,y1,z1),(x2,y2,z2))) # draw ray
                sectionpoints.append((x2,y2,z2))
        pp.addPoints(sectionpoints)
        #Points.show(pp) # draw intersection points per raybundle per field point

        return (pp, res)


    def makeRaysFromRayPath(self, raypath, offset, color = (0.5, 0.5, 0.5)):
        doc = FreeCAD.ActiveDocument # in initialisierung auslagern
        Nsurf = len(raypath.raybundles)
        offx = offset[0]
        offy = offset[1]
        offz = offset[2]

        for i in np.arange(Nsurf):
            offz += self.os.surfaces[i].getThickness()
            (intersectionpts, rays) = self.makeRayBundle(raypath.raybundles[i], offset=(offx, offy, offz))

            FCptsobj = doc.addObject("Points::Feature", "Surf_"+str(i)+"_Intersectionpoints")
            FCptsobj.Points = intersectionpts
            FCptsview = FCptsobj.ViewObject
            FCptsview.PointSize = 5.0
            FCptsview.ShapeColor = (1.0, 1.0, 0.0)

            self.intersectptsobs.append(FCptsobj)

            for (n, ray) in enumerate(rays):
                FCrayobj = doc.addObject("Part::Feature", "Surf_"+str(i)+"_Ray_"+str(n))
                FCrayobj.Shape = ray
                FCrayview = FCrayobj.ViewObject

                FCrayview.LineColor = color
                FCrayview.PointColor = (1.0, 1.0, 0.0)

                self.rayobs.append(FCrayobj)



    def showAimFiniteSurfaceStopDialog(self):

        pupiltype = core.pupil.EntrancePupilDiameter
        pupilsize = 2.0
        fieldType = core.field.ObjectHeight
        rasterType = core.raster.RectGrid
        stopPosition = 1


        ad = AimDialog(pupilsize, stopPosition)
        ad.exec_()
        if ad.pupiltype != "":
            pupiltype = eval("core.pupil."+ad.pupiltype) # eval is evil but who cares :p
        if ad.pupilsize != 0.0:
            pupilsize = ad.pupilsize
        if ad.fieldtype != "":
            fieldType = eval("core.field."+ad.fieldtype)
        if ad.rastertype != "":
            rasterType = eval("core.raster."+ad.rastertype)
        if ad.stopposition != 0:
            stopPosition = ad.stopposition


        res = (pupiltype, pupilsize, fieldType, rasterType, stopPosition)
        self.aimfinitestopdata = res
        self.aiminitialized = True # has to be performed at least one time
        return res

    def showFieldWaveLengthDialog(self):
        fieldvariables = [[0., 0.], [0., 3.0]]
        wavelength = 0.55

        fd = FieldDialog(fieldvariables, wavelength)
        fd.exec_()
        fieldvariables = fd.fieldpoints
        wavelength = fd.wavelength

        res = (fieldvariables, wavelength)
        self.fieldpoints = fieldvariables
        self.wavelength = wavelength
        self.fieldwaveinitialized = True # has to be performed at least one time
        return res

    def showSpotDiagrams(self, numrays):
        # todo: spotdiagramm in extra commando
        # todo: aktualisieren vom 3d layout
        # todo: so, dass man nach der optimierung gucken kann

        (pupiltype, pupilsize, fieldtype, rastertype, stopposition) = self.aimfinitestopdata


        aimy = core.aim.aimFiniteByMakingASurfaceTheStop(self.os, pupilType= pupiltype, \
                                                    pupilSizeParameter=pupilsize, \
                                                    fieldType= fieldtype, \
                                                    rasterType= rastertype, \
                                                    nray=numrays, wavelength=self.wavelength, \
                                                    stopPosition=stopposition)

        fig = plt.figure(1)

        numplots = len(self.fieldpoints)
        for index, fp in enumerate(self.fieldpoints):
            initialBundle = aimy.getInitialRayBundle(self.os, fieldXY=np.array(fp), wavelength=self.wavelength)
            rp = RayPath(initialBundle, self.os)
            ax = fig.add_subplot(numplots, 1, index)
            ax.axis('equal')


            core.plots.drawSpotDiagram(ax, self.os, rp, -1)
        plt.show()


    def createRayViews(self, numrays):

        if not self.aiminitialized or not self.fieldwaveinitialized:
            QtGui.QMessageBox.critical(None, "Pyrate", "Either aimy or field/wavelength matrix was not initialized. Please do that!")
            return


        doc = FreeCAD.ActiveDocument

        #numrays = 100
        #pupilsize = 2.0 # fix to appropriate system pupilsize
        #stopposition = 1 # fix to appropriate system stop position
        #wavelengthparam = 0.55

        #fieldvariable = [0., 0.]
        #fieldvariable2 = [0., 3.0]

        # (pupiltype, pupilsize, fieldType, rasterType, stopPosition)

        (pupiltype, pupilsize, fieldtype, rastertype, stopposition) = self.aimfinitestopdata


        aimy = core.aim.aimFiniteByMakingASurfaceTheStop(self.os, pupilType= pupiltype, \
                                                    pupilSizeParameter=pupilsize, \
                                                    fieldType= fieldtype, \
                                                    rasterType= rastertype, \
                                                    nray=numrays, wavelength=self.wavelength, \
                                                    stopPosition=stopposition)

        #aimy = core.aim.aimFiniteByMakingASurfaceTheStop(self.os, pupilType= core.pupil.EntrancePupilDiameter, \
        #                                            pupilSizeParameter=pupilsize, \
        #                                            fieldType= core.field.ObjectHeight, \
        #                                            rasterType= core.raster.RectGrid, \
        #                                            nray=numrays, wavelength=wavelengthparam, \
        #                                            stopPosition=stopposition)
        #aimy.setPupilRaster(rasterType= raster.ChiefAndComa, nray=numrays)
        #aimy.setPupilRaster(rasterType= raster.RectGrid, nray=numrays)

        #aimy.setPupilRaster(rasterType= core.raster.PoissonDiskSampling, nray=numrays)

        for fp in self.fieldpoints:
            initialBundle = aimy.getInitialRayBundle(self.os, fieldXY=np.array(fp), wavelength=self.wavelength)
            rp = RayPath(initialBundle, self.os)
            self.makeRaysFromRayPath(rp,offset=(0,0,0), color=(np.random.random(), np.random.random(), np.random.random()))


        for obj in doc.Objects:
            obj.touch()
        doc.recompute()



    def returnPrescriptionData(self):
        pass
        # get effective focal length and so on from self.os and put it into some string which could be shown by
        # some dialog


OSinterface = OpticalSystemInterface()






