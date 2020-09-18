#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPD-Reader demo

Copyright (C) 2020
               by     Ivo Ihrke
               
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
"""

import sys

#first add ../, then make sure it is first -- for local dev, uncomment next two lines
sys.path.append('../')
sys.path.reverse()

import pyrateoptics
from pyrateoptics import build_rotationally_symmetric_optical_system
from pyrateoptics import draw
from pyrateoptics import raytrace

from pyrateoptics.sampling2d.raster import RectGrid
from pyrateoptics.raytracer.ray import RayBundle, RayPath
from pyrateoptics.raytracer.material.material_glasscat import CatalogMaterial
from pyrateoptics.raytracer.globalconstants import canonical_ex, canonical_ey
from pyrateoptics.raytracer.localcoordinates import LocalCoordinates
from pyrateoptics.raytracer.analysis.ray_analysis import RayBundleAnalysis, RayPathAnalysis
from pyrateoptics.raytracer.io.spd import SPDParser


from matplotlib import pyplot as plt

import numpy as np


#ATTENTION: need to initialize submodules !
#> git submodule update --init --recursive
db_path = "../pyrateoptics/refractiveindex.info-database/database"

#this way, we may look for names
gcat = pyrateoptics.GlassCatalog(db_path)
#print( gcat.find_pages_with_long_name("BK7") )

if gcat.get_shelves() == []:
    print("""Get the glass data base first! 
             > git submodule update --init --recursive
             Be sure to run from the pyrate/demos directory.
          """);
    sys.exit( -1 );

options={'gcat' : gcat, \
         'db_path' : db_path };

spd = SPDParser( "data/Thorlabs_AC127_050_A.SPD", name="Thorlabs 50mm Doublet" );
#spd = SPDParser( "data/double_gauss_rudolph_1897_v2.SPD", name="rudolph2" );

(s,seq) = spd.create_optical_system(options = options);

draw(s)


def bundle(fpx, fpy, nrays=16, rpup=5):
    """
    Creates a RayBundle
    
    * o is the origin - this is ok
    * k is the wave vector, i.e. direction; needs to be normalized 
    * E0 is the polarization vector
    
    """

    (px, py) = RectGrid().getGrid(nrays)
    
    #pts_on_first_surf = np.vstack((rpup*px, rpup*py, np.zeros_like(px)))
    pts_on_entrance_pupil = np.vstack((rpup*px, rpup*py, np.ones(px.shape) * spd.psys.entpup) );
    
    o = np.concatenate([ fpx * spd.psys.field_size_obj() * np.ones([1,len(px)]), \
                        -fpy * spd.psys.field_size_obj() * np.ones([1,len(px)]), \
                         spd.psys.obj_dist() * np.ones([1,len(px)]) ], axis=0 );
    
    k = (pts_on_entrance_pupil - o); 
    
    normk = np.sqrt( [np.sum( k*k, axis=0 )] );
    k = k / ( np.matmul( np.ones([3,1]), normk ) );
    
    E0 = np.cross(k, canonical_ex, axisa=0, axisb=0).T

    #wavelength is in [mm]
    return RayBundle(o, k, E0, wave=0.00058756 ), px, py


def meridional_bundle( fpy, nrays=16, rpup = 5 ):
    """
    Creates a RayBundle
    
    * o is the origin - this is ok
    * k is the wave vector, i.e. direction; needs to be normalized 
    * E0 is the polarization vector
    
    """

    pts_on_entrance_pupil = np.vstack((np.zeros([1,nrays]), \
                                       np.linspace(rpup,-rpup,nrays)[None,...], \
                                       np.ones([1,nrays]) * spd.psys.entpup) );
    
    o = np.concatenate([ np.zeros([1,nrays]), \
                        -fpy * spd.psys.field_size_obj() * np.ones([1,nrays]), \
                         spd.psys.obj_dist() * np.ones([1,nrays]) ], axis=0 );
    
    k = (pts_on_entrance_pupil - o); 
    normk = np.sqrt( [np.sum( k*k, axis=0 )] );
    k = k / ( np.matmul( np.ones([3,1]), normk ) );

    E0 = np.cross(k, canonical_ex, axisa=0, axisb=0).T

    #wavelength is in [mm]
    return RayBundle(o, k, E0, wave=0.00058756 )


    
    
#This is for comparison with WinLens: Tables -> RayFan 
#fixes intersection position to 1e-7; 
#b1 = meridional_bundle( 0.0,  nrays=15, rpup = psys.entpup_rad - 0.0000695 ); #AC127_050_A
#b1 = meridional_bundle( 0.0,  nrays=15, rpup = psys.entpup_rad + 0.0087 );
#b2 = meridional_bundle(-1.0,  nrays=15, rpup = psys.entpup_rad );

b1, px, py = bundle( 0.0, 0.0,  nrays=128, rpup = spd.psys.entpup_rad - 0.0000695 );
b2, px, py = bundle( 0.0, 1.0,  nrays=128, rpup = spd.psys.entpup_rad );


plt.clf();
r1 = s.seqtrace(b1, seq)
r2 = s.seqtrace(b2, seq)

draw(s, r1) # show system + rays
draw(s, r2) # show system + rays

#* r1 and r2 are lists of class RayPath (in raytacer/ray.py)
#  - we can analyze them with the class RayPathAnalysis (in raytracer/ray_analysis.py)
#
#* r1 and r2 are lists of RayPaths (of length 1), RayPaths contain RayBundles, a RayBundle is the intersection with one surface
#  - actually a RayBundle can also describe ray segments
#  - let rb be a ray bundle, it contains variables rb.x (the intersection positions) and rb.k (the Pointing vectors)
#  - the shape of rp.x can be (from what I've seen) [1,3,N] - in this case, the rb describes an intersection with a surface
#    or                                             [2,3,N] - in this case, the rb describes a ray segment
#
#* the seqtrace command returns a list of raybundles of the ray segment type
#  - r1[0].raybundles[0].x.shape should give [2,3,N], similar to r1[0].raybundles[1].x.shape, ...
#  - len(r1[0].raybundles) yields the number of ray segments 

#a ray bundle analysis can analyze a ray distribution at a particular surface
ra=RayBundleAnalysis(r2[0].raybundles[-1],'R2')
print( ra.get_centroid_position() )
print( ra.get_rms_spot_size_centroid() )


from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
fig = plt.gcf()
fig.clf();
ax = fig.add_subplot(111, projection='3d')

#a ray path analysis analyzes the full length of the ray trajectories, again for 
#a bundle of rays; 
#* to me, the terminology is confusing: I would expect a RayPath to be that of a single ray, whereas a Raybundle is that of many rays; but this is not the case!
#

#compute OPL for the full path up to the image plane
#- the first raybundle in the RayPath is 'segment type', it is a duplicate of the first segment, i,e. raybundles[0]==raybundles[1] always ?
#- the last raybundle is a repeated 'position type'; the location is already in the previous segment (as its second point)
raypath = RayPath( r1[0].raybundles[1:-1] )
rpa = RayPathAnalysis(raypath);
opl = rpa.get_optical_path_length();
#I verified the OPL along the optical axis of the test system; it appears to be ok. 

#compute the OPL only up to the surface of the system stop
# - create a sub-RayPath and run analysis on this one
stop_index = spd.get_stop_surface_index()
rpa_stop = RayPathAnalysis(RayPath( raypath.raybundles[0:stop_index+1] ) )
opl_stop = rpa_stop.get_optical_path_length();

# now, back-propagate the rays in the image plane to the exit pupil
# - by subtracting the OPL difference 

#this is positive
opl_diff = opl - opl_stop 
#the last RayBundle contains the intersection with the image plane
# - the last RayBundle is a 'position type' bundle, we need to pick the first (and only) component
pts_imgpl = raypath.raybundles[-1].x[0,:,:]
dir_imgpl = raypath.raybundles[-1].k[0,:,:]
refindex  = np.sqrt(np.sum(np.abs(dir_imgpl)**2.0, axis=0))
#normalize the Pointing vectors
dir_imgpl = np.real(dir_imgpl) / np.matmul( np.ones([3,1]), [refindex] )

#re-base the image plane to the nominal value from the optical system (that is 
#measured w.r.t. the last system surface, whereas the ray tracing values are 
#cumulative)
pts_imgpl[2,:] = spd.psys.img_dist()
#this propagates backward because opl_diff is positive
wavefront_pts = pts_imgpl - np.matmul( np.ones([3,1]), [opl_diff] ) * dir_imgpl

#!!! the calculation is off by ~3.65mm (for rudolph), this is the size of the double air gap (each part) around the stop
#!!! - maybe this has something to do with it
#!!! - tried Thorlabs_AC127_050; there is also a difference
#!!! - the on-axis OPL seems to be ok (object-image)

#for interpretation, need position of last surface --> not necessary with above re-base
# - this can be obtained from the image distance of the system (which is measured w.r.t. the last surface of the system)
#wavefront_pts[2,:] = spd.psys.img_dist() - wavefront_pts[2,:];

#surf=ax.plot_trisurf(px,py,opl-np.amin(opl),cmap=cm.jet)
surf=ax.plot_trisurf(wavefront_pts[0,:], wavefront_pts[1,:], wavefront_pts[2,:], cmap=cm.jet)
fig.colorbar(surf, shrink=0.5, aspect=5);

