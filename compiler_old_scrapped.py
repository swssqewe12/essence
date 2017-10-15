import os

###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class Parser(object):
    def __init__(self, data):
        self.data = data
        self.index_list = [0]
        self.index = self.index_list[0]
        self.peeking_list = [False]
        self.peeking = self.peeking_list[0]
        self.char = data[0]

    def next(self):
        self.index += 1
        self.char = self.data[self.index]

    def peek(self):
        self.index_list.append(self.index_list[-1])
        self.index = self.index_list[-1]
        self.peeking_list.append(True)
        self.peeking = self.peeking_list[-1]

    def peek_done(self, success=False):
        dif = self.index_list[-1] - self.index_list[-2]
        del self.index_list[-1]
        del self.peeking_list[-1]
        self.peeking = self.peeking_list[-1]
        if success:
            self.index_list[-1] += dif
        self.index = self.index_list[-1]

    def syntax_error(self):
        if self.peeking:
            return None
        raise_error("main.ess", "Invalid syntax", self.data, self.index)

    def parse(self):
        return self.program()
        
    ##############################
    # TOKENS                     #
    ##############################

    ##############################
    # RULES                      #
    ##############################

    def program(self):
        func_defs = self.func_defs()
        return {'func_defs': func_defs}

    def func_defs(self):
        while True:
            self.func_def();
            self.peek()
            func_type, func_name = self.func_def_a()


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

#####################################
# Base NodeVisitor                  #
#####################################

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

#####################################
# Interpreter                       #
#####################################

class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def interpret(self):
        tree = self.parser.parse()
        if tree is None:
            return ''
        return self.visit(tree)

    #####################
    # Visitors          #
    #####################

    def visit_NumberNode(self, node):
        return node.value

    def visit_BinOpNode(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == CARET:
            return self.visit(node.left) ** self.visit(node.right)

    def visit_UnaryOpNode(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)

###############################################################################
#                                                                             #
#  FUNCTIONS                                                                  #
#                                                                             #
###############################################################################

def get_line_col(string, index):
    cur_pos = 0
    lines = string.splitlines(True)
    for linenum, line in enumerate(lines):
        if cur_pos + len(line) > index:
            return linenum + 1, index-cur_pos
        cur_pos += len(line)

def get_line(string, line):
    return string.split("\n")[line]

def raise_simple_error(f, error):
    os.system("cls")
    print "An error has occured in `" + f + "`"
    print "Error: " + error
    print
    os.system("pause")
    exit(0)

def raise_error(f, error, string, index):
    line, col = get_line_col(string, index)
    os.system("cls")
    print "An error has occured in `" + f + "`"
    print
    print "Error: " + error
    print "Ln: " + str(line) + "  Col: " + str(col)
    print
    print get_line(string, line-1)
    print " " * col + "^"
    print
    os.system("pause")
    exit(0)

###############################################################################
#                                                                             #
#  PROGRAM STARTS HERE                                                        #
#                                                                             #
###############################################################################

if __name__ == '__main__':
        
    f = open("project/src/main.ess", "r")
    data = f.read()
    f.close()

    if len(data) == 0:
        raise_simple_error("main.ess", "Empty file")
        
    parser = Parser(data)
    result = parser.parse()
    
    import json
    print json.dumps(result, default=lambda o: o.__dict__)
    print

    os.system("pause")
