# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 07:51:00 2022

@author: smanigand
"""

import os
from PyRANA_v1 import PyRANA


filename = "210212_003_532_PYX110_spot3.txtr.meta"


pyrana = PyRANA()

pyrana.loadfile_METAFILE(os.path.join("example", filename))



pyrana.plot_all()



fit = pyrana.quickfit(Xmin=880, Xmax=1080, continuum='poly2', 
                     peaks=[{'profile': 'cauchy', 'x0': 930},
                            {'profile': 'cauchy', 'x0': 1005,},
                            {'profile': 'cauchy', 'x0': 1025,},
                            ])

fit2 = pyrana.quickfit(Xmin=630, Xmax=800, continuum='poly2', 
                     peaks=[{'profile': 'cauchy', 'x0': 660},
                            {'profile': 'cauchy', 'x0': 680,},
                            {'profile': 'cauchy', 'x0': 730,},
                            ])

fit3 = pyrana.quickfit(Xmin=260, Xmax=480, continuum="linear",
                       peaks=[{'profile': 'assymCauchy', 'x0': 335},
                            #{'profile': 'assymCauchy', 'x0': 375,},
                            {'profile': 'cauchy', 'x0': 395,},
                            {'profile': 'cauchy', 'x0': 410,}
                            ])

fit4 = pyrana.quickfit(Xmin=180, Xmax=260, continuum="poly2",
                       peaks=[{'profile': 'assymCauchy', 'x0': 230}])

pyrana.plot_fit(fit)

pyrana.plot_fit(fit2)

pyrana.plot_fit(fit3)

pyrana.plot_fit(fit4)

pyrana.plot_all([fit, fit2, fit3, fit4])


