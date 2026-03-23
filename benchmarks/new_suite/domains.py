from lappy.geometry import (eq_tri, iso_right_tri, iso_tri, rect, parallelogram, disk_sector, L_shape,
                            GWW1, GWW2, H_shape, chevron, mushroom, cut_square)

import numpy as np

# domains
# triangles
eq_tri = eq_tri()
iso_right_tri = iso_right_tri()
iso_tri_05 = iso_tri(0.5)
iso_tri_1 = iso_tri(1.0)
iso_tri_2 = iso_tri(2.0)
iso_tri_4 = iso_tri(4.0)
iso_tri_8 = iso_tri(8.0)
iso_tri_16 = iso_tri(16.0)

# convex quadrilaterals
square = rect(1, 1)
rect_1_pi = rect(1, np.pi)
rect_1_20 = rect(1, 20)
para_pi_3 = parallelogram(1, 1, np.pi/3)
para_1 = parallelogram(1, 1, 1.0)

# disk sectors
sector_pi_8 = disk_sector(theta=np.pi/8)
sector_05 = disk_sector(theta=0.5)
sector_pi_4 = disk_sector(theta=np.pi/4)
sector_1 = disk_sector(theta=1.0)
sector_pi_2 = disk_sector(theta=np.pi/2)
sector_pi = disk_sector(theta=np.pi)
sector_2pi_3 = disk_sector(theta=2*np.pi/3)
sector_7pi_8 = disk_sector(theta=7*np.pi/8)

# polygons with reentrant corners
lshape = L_shape()
gww1 = GWW1()
gww2 = GWW2()
hshape = H_shape()
chevron_12 = chevron(1,2)
chevron_14 = chevron(1,4)
chevron_34 = chevron(3,4)

# curved boundary segments with corners
mushroom = mushroom()
cut_square_025 = cut_square(0.25)
cut_square_05 = cut_square(0.5)
cut_square_075 = cut_square(0.75)