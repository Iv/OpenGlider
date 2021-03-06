import copy
import numpy

from openglider.glider.rib.elements import AttachmentPoint, CellAttachmentPoint
from openglider.lines import Node, Line, LineSet
from openglider.utils import recursive_getattr
from openglider.lines import line_types


class LowerNode2D(object):
    """lower attachment point"""
    def __init__(self, pos_2D, pos_3D, name="unnamed", layer=None):
        self.pos_2D = pos_2D
        self.pos_3D = pos_3D
        self.name = name
        self.layer = layer or ""

    def __json__(self):
        return{
            "pos_2D": self.pos_2D,
            "pos_3D": self.pos_3D,
            "name": self.name,
            "layer": self.layer}

    def get_2D(self, *args):
        return self.pos_2D

    def get_node(self, glider):
        return Node(node_type=0, position_vector=numpy.array(self.pos_3D))


class UpperNode2D(object):
    """stores the 2d data of an attachment point"""
    def __init__(self, cell_no, rib_pos, cell_pos=0, force=1., name="unnamed", layer=None):
        self.cell_no = cell_no
        self.cell_pos = cell_pos
        self.rib_pos = rib_pos  # value from 0...1
        self.force = force
        self.name = name
        self.layer = layer or ""

    def __json__(self):
        return {'cell_no': self.cell_no,
                'rib_pos': self.rib_pos,
                "cell_pos": self.cell_pos,
                'force': self.force,
                'name': self.name,
                "layer": self.layer}

    def get_2D(self, parametric_shape):
        return parametric_shape[self.cell_no, self.rib_pos]

    def get_node(self, glider):
        if self.cell_pos > 0:
            cell = glider.cells[self.cell_no + glider.has_center_cell]
            if isinstance(self.force, (list, tuple, numpy.ndarray)):
                force = list(self.force)
            else:
                midrib = cell.midrib(self.cell_pos)
                force1 = numpy.array([0, self.force, 0])
                plane = midrib.projection_layer
                force = list(numpy.array(plane.translation_matrix.dot(force1))[0])

            node = CellAttachmentPoint(cell, self.name, self.cell_pos, self.rib_pos, force)
        else:
            rib = glider.ribs[self.cell_no + glider.has_center_cell]
            if isinstance(self.force, (list, tuple, numpy.ndarray)):
                force = list(self.force)
            else:
                force = list(rib.rotation_matrix.dot(numpy.array([0, self.force, 0])))
            node = AttachmentPoint(glider.ribs[self.cell_no + glider.has_center_cell],
                                   self.name, self.rib_pos, force)


        node.get_position()
        return node


class BatchNode2D(object):
    def __init__(self, pos_2D, name=None, layer=None):
        self.pos_2D = pos_2D  # pos => 2d coordinates
        self.name = name
        self.layer = layer or ""

    def __json__(self):
        return{
            "pos_2D": self.pos_2D,
            "name": self.name,
            "layer": self.layer
        }

    def get_node(self, glider):
        return Node(node_type=1)

    def get_2D(self, *args):
        return self.pos_2D


class LineSet2D(object):
    def __init__(self, line_list):
        self.lines = line_list

    def __json__(self):
        lines = [copy.copy(line) for line in self.lines]
        nodes = self.nodes
        for line in lines:
            line.upper_node = nodes.index(line.upper_node)
            line.lower_node = nodes.index(line.lower_node)
        return {"lines": lines,
                "nodes": nodes}

    @classmethod
    def __from_json__(cls, lines, nodes):
        lineset = cls(lines)
        for line in lineset.lines:
            if isinstance(line.upper_node, int):
                line.upper_node = nodes[line.upper_node]
            if isinstance(line.lower_node, int):
                line.lower_node = nodes[line.lower_node]
        return lineset

    @property
    def nodes(self):
        nodes = set()
        for line in self.lines:
            nodes.add(line.upper_node)
            nodes.add(line.lower_node)

        return list(nodes)

    def get_upper_nodes(self, rib_no=None):
        nodes = set()
        for line in self.lines:
            node = line.upper_node
            if isinstance(node, UpperNode2D):
                if rib_no is None or node.cell_no == rib_no:
                    nodes.add(line.upper_node)

        return list(nodes)

    def get_upper_node(self, name):
        for node in self.get_upper_nodes():
            if node.name == name:
                return node

    def get_lower_attachment_points(self):
        return [node for node in self.nodes if isinstance(node, LowerNode2D)]

    def return_lineset(self, glider, v_inf):
        """
        Get Lineset_3d
        :param glider: Glider3D
        :param v_inf:
        :return: LineSet (3d)
        """
        #v_inf = v_inf or glider.v_inf
        lines = []
        # now get the connected lines
        # get the other point (change the nodes if necesarry)
        for node in self.get_lower_attachment_points():
            self.sort_lines(node)
        self.delete_not_connected(glider)

        nodes_3d = {node: node.get_node(glider) for node in self.nodes}
        # set up the lines!
        for line_no, line in enumerate(self.lines):
            lower = nodes_3d[line.lower_node]
            upper = nodes_3d[line.upper_node]
            if lower and upper:
                line = Line(number=line_no, lower_node=lower, upper_node=upper,
                            v_inf=None, target_length=line.target_length,
                            line_type=line.line_type, name=line.name)
                lines.append(line)

        return LineSet(lines, v_inf)

    def scale(self, factor):
        for line in self.lines:
            target_length = getattr(line, "target_length", None)
            if target_length is not None:
                line.target_length *= factor
        for node in self.get_lower_attachment_points():
            node.pos_3D = numpy.array(node.pos_3D) * factor
            node.pos_2D = numpy.array(node.pos_2D) * factor

    def set_default_nodes2d_pos(self, glider):
        lineset_3d = self.return_lineset(glider, [10,0,0])
        lineset_3d._calc_geo()
        line_dict = {line_no: line2d for
                     line_no, line2d in enumerate(self.lines)}

        for line in lineset_3d.lines:
            pos_3d = line.upper_node.vec
            pos_2d = [pos_3d[1], pos_3d[2]]
            line_dict[line.number].upper_node.pos_2D = pos_2d

    def sort_lines(self, lower_att):
        """
        Recursive sorting of lines (check direction)
        """
        for line in self.lines:
            if not line.is_sorted:
                if lower_att == line.upper_node:
                    line.lower_node, line.upper_node = line.upper_node, line.lower_node
                if lower_att == line.lower_node:
                    line.is_sorted = True
                    self.sort_lines(line.upper_node)

    def get_upper_connected_lines(self, node):
        return [line for line in self.lines if line.lower_node is node]

    def get_lower_connected_lines(self, node):
        return [line for line in self.lines if line.upper_node is node]

    def get_influence_nodes(self, line):
        if isinstance(line.upper_node, UpperNode2D):
            return [line.upper_node]
        return sum([self.get_influence_nodes(l) for l in self.get_upper_connected_lines(line.upper_node)], [])

    def create_tree(self, start_node=None):
        """
        Create a tree of lines
        :return: [(line, [(upper_line1, []),...]),(...)]
        """
        if start_node is None:
            start_node = self.get_lower_attachment_points()
            lines = []
            for node in start_node:
                lines += self.get_upper_connected_lines(node)
        else:
            lines = self.get_upper_connected_lines(start_node)

        def get_influence_nodes(line):
            if isinstance(line.upper_node, UpperNode2D):
                return [line.upper_node]
            return sum([get_influence_nodes(l) for l in self.get_upper_connected_lines(line.upper_node)], [])

        for line in lines:
            if not get_influence_nodes(line):
                return line

        def sort_key(line):
            nodes = get_influence_nodes(line)
            if not nodes:
                print("line", line)
                #return -1
            return sum([100*node.cell_no+node.rib_pos for node in nodes])/len(nodes)

        lines.sort(key=sort_key)

        return [(line, self.create_tree(line.upper_node)) for line in lines]

    def delete_not_connected(self, glider):
        temp = []
        temp_new = []
        for line in self.lines:
            if isinstance(line.upper_node, UpperNode2D):
                if line.upper_node.cell_no >= len(glider.ribs):
                    temp.append(line)
                    self.nodes.remove(line.upper_node)

        while temp:
            for line in temp:
                conn_up_lines = [j for j in self.lines if (j.lower_node == line.lower_node and j != line)]
                conn_lo_lines = [j for j in self.lines if (j.upper_node == line.lower_node and j != line)]
                if len(conn_up_lines) == 0:
                    self.nodes.remove(line.lower_node)
                    self.lines.remove(line)
                    temp_new += conn_lo_lines
                temp.remove(line)
            temp = temp_new


class Line2D(object):
    def __init__(self, lower_node, upper_node, 
                 target_length=None, line_type='default', layer=None, name=None):
        self.lower_node = lower_node
        self.upper_node = upper_node
        self.target_length = target_length
        self.is_sorted = False
        self.line_type = line_types.LineType.get(line_type)
        self.layer = layer or ""
        self.name = name


    def __json__(self):
        return{
            "lower_node": self.lower_node,
            "upper_node": self.upper_node,
            "target_length": self.target_length,
            "line_type": self.line_type.name,
            "layer": self.layer,
            "name": self.name
            }