#! /usr/bin/python2
# -*- coding: utf-8; -*-
#
# (c) 2013 booya (http://booya.at)
#
# This file is part of the OpenGlider project.
#
# OpenGlider is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# OpenGlider is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenGlider.  If not, see <http://www.gnu.org/licenses/>.
import unittest
import sys
import os

from PyQt4 import QtGui
from openglider.gui import ApplicationWindow


try:
    import openglider
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import openglider

from openglider.input import ControlPoint, MplWidget
from openglider.input.ballooning import input_ballooning
from openglider.input.shape import shapeinput, MplSymmetricBezier

qApp = QtGui.QApplication(sys.argv)
testfolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
importpath = testfolder + '/demokite.ods'


class GliderTestClass(unittest.TestCase):
    def setUp(self, complete=True):
        self.glider = openglider.glider.Glider.import_geometry(path=importpath)

    def test_spline_input(self):
        points = [[.1, .2], [.2, .2], [.3, .6], [.6, .0]]
        controlpoints = [ControlPoint(p, locked=[0, 0]) for p in points]
        # print(mpl1)
        line1 = MplSymmetricBezier(controlpoints)  #, mplwidget=mpl1)
        mplwidget = MplWidget(dpi=100)
        line1.insert_mpl(mplwidget)
        aw = ApplicationWindow([mplwidget])
        mplwidget.redraw()
        aw.show()
        qApp.exec_()

    def test_shape_input(self):
        window = shapeinput(self.glider)
        window.show()
        qApp.exec_()

    def test_ballooning_input(self):
        ballooning = self.glider.ribs[0].ballooning
        window = input_ballooning(ballooning)
        window.show()
        qApp.exec_()

if __name__ is '__main__':
    unittest.main()