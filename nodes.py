class Node(object):
    """Base class for all nodes
    """
    _id_counter = 0

    def __init__(self):
        self.id = Node._id_counter
        Node._id_counter += 1

class OrNode(Node):
    """Class for or nodes
    """
    _node_type = "or"

    def __init__(self):
        Node.__init__(self)
        self.left_child = None
        self.right_child = None
        self.left_weight = 0.0
        self.right_weight = 0.0

class AndNode(Node):
    """Class for and nodes
    """
    _node_type = "and"

    def __init__(self):
#        Node.__init__(self, var_scope)
        self.children_left = None
        self.children_right = None
        self.or_features = None
        self.left_weights = None
        self.right_weights = None
        self.forest = None
        self.roots = None
        self.tree_forest = None
        self.cltree = None

class TreeNode(Node):
    """Class for tree nodes
    """
    _node_type = "tree"

    def __init__(self):
#        Node.__init__(self, var_scope)
        self.cltree = None
        self.or_feature = None

###############################################################################

def is_or_node(node):
    """Returns True if the given node is a or node."""
    return getattr(node, "_node_type", None) == "or"

def is_and_node(node):
    """Returns True if the given node is a and node."""
    return getattr(node, "_node_type", None) == "and"

def is_tree_node(node):
    """Returns True if the given node is a tree node."""
    return getattr(node, "_node_type", None) == "tree"

