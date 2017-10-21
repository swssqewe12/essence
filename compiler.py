###############################################################################
#                                                                             #
#  IMPORTS                                                                    #
#                                                                             #
###############################################################################

import os, string, json

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

#####################################
# Tokens                            #
#####################################

IDENTIFIER  =   'IDENTIFIER'
INTEGER     =   'INTEGER'
FLOAT       =   'FLOAT'
LPAREN      =   'LPAREN'
RPAREN      =   'RPAREN'
LBRACE      =   'LBRACE'
RBRACE      =   'RBRACE'
EQUALS      =   'EQUALS'
SEMI        =   'SEMI'
PLUS        =   'PLUS'
MINUS       =   'MINUS'
MUL         =   'MUL'
DIV         =   'DIV'
CARET       =   'CARET'
COMMA       =   'COMMA'
EOF         =   'EOF'

class Token(object):
    def __init__(self, typ, val=""):
        self.type = typ
        self.value = val
        self.pos = lexer.pos
    def val(self, val):
        self.value = val
    

#####################################
# Lexer                             #
#####################################

class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise_error("main.ess", "Illegal character", self.text, self.pos)

    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def identifier(self):

        letters = list(string.ascii_uppercase + string.ascii_lowercase)
        digits = map(lambda x:str(x), list(range(10)))

        result = ""

        while self.current_char in letters + digits:
            result += self.current_char
            self.advance()

        return result

    def number(self):

        has_decimal_point = False
        digits = map(lambda x:str(x), list(range(10))) + ['.']

        result = ""

        while self.current_char in digits:
            result += self.current_char
            if self.current_char == '.':
                if not has_decimal_point:
                    has_decimal_point = True
                else:
                    raise_error("main.ess", "Failed to create token. Float can not have more than one decimal point.", self.text, self.pos)
            self.advance()

        return result, has_decimal_point  

    def get_next_token(self):
        
        while self.current_char is not None:

            letters = list(string.ascii_uppercase + string.ascii_lowercase)
            digits = map(lambda x:str(x), list(range(10)))

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '(':
                token = Token(LPAREN, '(')
                self.advance()
                return token

            if self.current_char == ')':
                token = Token(RPAREN, ')')
                self.advance()
                return token

            if self.current_char == '{':
                token = Token(LBRACE, '{')
                self.advance()
                return token

            if self.current_char == '}':
                token = Token(RBRACE, '}')
                self.advance()
                return token

            if self.current_char == '=':
                token = Token(EQUALS, '=')
                self.advance()
                return token

            if self.current_char == ';':
                token = Token(SEMI, ';')
                self.advance()
                return token

            if self.current_char == '+':
                token = Token(PLUS, '+')
                self.advance()
                return token

            if self.current_char == '-':
                token = Token(MINUS, '-')
                self.advance()
                return token

            if self.current_char == '*':
                token = Token(MUL, '*')
                self.advance()
                return token

            if self.current_char == '/':
                token = Token(DIV, '/')
                self.advance()
                return token

            if self.current_char == '^':
                token = Token(CARET, '^')
                self.advance()
                return token

            if self.current_char == ',':
                token = Token(COMMA, ',')
                self.advance()
                return token

            if self.current_char in letters:
                token = Token(IDENTIFIER)
                token.val(self.identifier())
                return token

            if self.current_char in digits:
                float_token = Token(FLOAT)
                int_token   = Token(INTEGER)
                
                number, is_float = self.number()
                
                if is_float:
                    float_token.val(number)
                    return float_token
                else:
                    int_token.val(number)
                    return int_token

            self.error()

        return Token(EOF, None)

###############################################################################
#                                                                             #
#  NODES AND NODE VISITOR                                                     #
#                                                                             #
###############################################################################

class Node(object):
    pass

class NodeVisitor(object):
    def visit(self, node, *args):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, *args)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

#####################################

class Program(Node):
    def __init__(self, decls, assignments):
        self.decls = decls
        self.assignments = assignments
        
class FunctionDeclarationNode(Node):
    def __init__(self, typ, name, args, decls, statements):
        self.type_tok = typ
        self.name_tok = name
        self.args = args
        self.decls = decls
        self.statements = statements

class VariableDeclarationNode(Node):
    def __init__(self, typ, name):
        self.type_tok = typ
        self.name_tok = name

class ExpressionNode(Node):
    def __init__(self, node, pos):
        self.node = node
        self.type = None
        self.is_constant = None
        self.tok_pos = pos
    def set_type(self, typ):
        self.type = typ
    def set_constant(self, is_constant):
        self.is_constant = is_constant

class NumberNode(Node):
    def __init__(self, token):
        self.token = token

class BinOpNode(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op_tok = op
        self.right = right

class UnaryOpNode(Node):
    def __init__(self, op, node):
        self.op_tok = op
        self.node = node

class VariableNode(Node):
    def __init__(self, token):
        self.name_tok = token

class AssignmentNode(Node):
    def __init__(self, name_tok, expr):
        self.name_tok = name_tok
        self.expr = expr

###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise_error("main.ess", "Invalid syntax", self.lexer.text, self.current_token.pos)

    def tryeat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
            return True
        return False

    def eat(self, *token_types):
        value = self.current_token
        if not token_types:
            self.current_token = self.lexer.get_next_token()
            return value
        error = True
        for token_type in token_types:
            if self.tryeat(token_type):
                error = False
                break
        if error:

            self.error()
        return value

    def token_is(self, *token_types):
        return self.current_token.type in token_types

    def parse(self):
        tree = self.program()
        if not self.token_is(EOF):
            self.error()
        return tree

    #####################
    # Rules             #
    #####################

    def program(self):
        decls, assignments = self.declarations()
        return Program(decls, assignments)

    def declarations(self):
        decls = []
        assignments = []
        while self.token_is(IDENTIFIER):
            decl, assignment = self.declaration()
            decls += decl
            assignments += assignment
        return decls, assignments

    def declaration(self):
        decls = []
        assignments = []
        typ = self.eat(IDENTIFIER)

        def addvar():
            decls.append(VariableDeclarationNode(typ, name))
            if expr:
                assignments.append(AssignmentNode(name, expr))

        while True:

            name = self.eat(IDENTIFIER)

            if self.tryeat(EQUALS):
                expr = self.expression()
            else:
                expr = None

            if self.tryeat(COMMA):
                addvar()
                continue

            if self.tryeat(SEMI):
                addvar()
                break

            args = self.function_definition_argument_list()
            scope_decls, statements = self.compound_statement()

            decls.append(FunctionDeclarationNode(typ, name, args, scope_decls, statements))

            if self.tryeat(COMMA):
                continue

            break

        return decls, assignments

    def function_definition_argument_list(self):
        self.eat(LPAREN)
        self.eat(RPAREN)
        return []

    def compound_statement(self):
        decls = []
        statements = []
        self.eat(LBRACE)
        while not self.token_is(RBRACE):
            decl, assignment = self.declaration()
            decls += decl
            statements += assignment
        self.eat(RBRACE)
        return decls, statements

    def statement(self):
        return

    def argument_list(self):
        self.eat(LPAREN)
        self.eat(RPAREN)
        return []

    def factor(self):
        "factor : (PLUS | MINUS) factor | INTEGER | FLOAT | IDENTIFIER | LPAREN expr RPAREN"
        
        token = self.current_token

        if self.token_is(PLUS, MINUS):
            self.eat()
            node = UnaryOpNode(token, self.factor())
            return node
        
        elif self.tryeat(INTEGER) or self.tryeat(FLOAT):
            return NumberNode(token)

        elif self.tryeat(IDENTIFIER):
            # currently assuming this is a variable not a function call
            return VariableNode(token)
        
        elif self.tryeat(LPAREN):
            node = self.expr()
            self.eat(RPAREN)
            return node

    def power(self):
        "power : factor (CARET factor)*"

        node = self.factor()

        while self.token_is(CARET):

            token = self.current_token
            self.eat()
            node = BinOpNode(left=node, op=token, right=self.factor())

        return node

    def term(self):
        "term : power ((MUL | DIV) power)*"

        node = self.power()

        while self.token_is(MUL, DIV):

            token = self.current_token
            self.eat()
            node = BinOpNode(left=node, op=token, right=self.power())

        return node

    def expression(self):
        pos = self.current_token.pos
        return ExpressionNode(self.expr(), pos)

    def expr(self):
        "expr : term ((PLUS | MINUS) term)*"
        
        node = self.term()

        while self.token_is(PLUS, MINUS):
            
            token = self.current_token
            self.eat()
            node = BinOpNode(left=node, op=token, right=self.term())

        return node

###############################################################################
#                                                                             #
#  SYMBOLS                                                                    #
#                                                                             #
###############################################################################

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
        self.scope_level = 0

    def set_parent(self, parent):
        self.parent = parent
        self.scope_level = parent.scope_level + 1
    
    def add(self, symbol):
        self.symbols[symbol.name] = symbol

    def has(self, name):
        return name in self.symbols

    def get(self, name):
        return self.symbols.get(name, None)

    def get_global(self, name):
        result = self.get(name)
        if result == None and self.parent != None:
            return self.parent.get_global(name)
        return result

    def has_global(self, name):
        if self.has(name):
            return True
        if self.parent != None:
            return self.parent.has_global(name)
        return False

class Symbol(object):
    def __init__(self, name, typ=None):
        self.name = name
        self.type = typ

class VarSymbol(Symbol):
    def __init__(self, name, typ):
        Symbol.__init__(self, name, typ)

class BuiltInTypeSymbol(Symbol):
    def __init__(self, name):
        Symbol.__init__(self, name)

class FunctionSymbol(Symbol):
    def __init__(self, name, typ):
        Symbol.__init__(self, name, typ)

TYPE_VOID   = BuiltInTypeSymbol("void")
TYPE_INT    = BuiltInTypeSymbol("int")
TYPE_FLOAT  = BuiltInTypeSymbol("float")

###############################################################################
#                                                                             #
#  SYMBOL TABLE BUILDER                                                       #
#                                                                             #
###############################################################################

class SymbolTableBuilder(NodeVisitor):

    def visit_Program(self, program):
        program.symbol_table = SymbolTable()
        program.symbol_table.add(TYPE_VOID)
        program.symbol_table.add(TYPE_INT)
        program.symbol_table.add(TYPE_FLOAT)

        for decl in program.decls:
            self.visit(decl, program.symbol_table)

    def visit_FunctionDeclarationNode(self, func, parent_table):

        parent_table.add(FunctionSymbol(func.name_tok.value, parent_table.get_global(func.type_tok.value)))

        func.symbol_table = SymbolTable()
        func.symbol_table.set_parent(parent_table)

        for decl in func.decls:
            self.visit(decl, func.symbol_table)

    def visit_VariableDeclarationNode(self, var, parent_table):

        parent_table.add(VarSymbol(var.name_tok.value, parent_table.get_global(var.type_tok.value)))

###############################################################################
#                                                                             #
#  SEMANTIC ANALYZER                                                          #
#                                                                             #
###############################################################################

class SemanticAnalyzer(NodeVisitor):

    def __init__(self):
        self.expr_type_analyzer = ExpressionTypeAnalyzer()
    
    def visit_Program(self, program):
        
        for decl in program.decls:
            self.visit(decl, program.symbol_table)

        for assignment in program.assignments:
            self.visit(assignment, program.symbol_table)

    def visit_FunctionDeclarationNode(self, func, parent_table):
        self.visit_FunctionDeclarationNode_or_VariableDeclarationNode(func, parent_table)

        for decl in func.decls:
            if not isinstance(decl, VariableDeclarationNode):
                raise_error("main.ess", "Cannot have function declaration in other function", data, decl.name_tok.pos)

        for statement in func.statements:
            self.visit(statement, func.symbol_table)

    def visit_VariableDeclarationNode(self, decl, parent_table):
        self.visit_FunctionDeclarationNode_or_VariableDeclarationNode(decl, parent_table)

    def visit_AssignmentNode(self, assignment, parent_table):
        self.visit(assignment.expr, parent_table)
        symbol = parent_table.get_global(assignment.name_tok.value)

        if symbol.type != assignment.expr.type:
            raise_error("main.ess", "Variable with type `" + symbol.type.name + "` cannot be assigned to type `" + assignment.expr.type.name + "`", data, assignment.expr.tok_pos)

    def visit_FunctionDeclarationNode_or_VariableDeclarationNode(self, decl, parent_table):
        symbol = parent_table.get_global(decl.type_tok.value)
        
        if symbol == None:
            raise_error("main.ess", "Symbol `" + decl.type_tok.value + "` not found", data, decl.type_tok.pos)
        
        if not isinstance(symbol, BuiltInTypeSymbol):
            raise_error("main.ess", "Expected built-in type but instead got " + get_symbol_type(symbol) + " `" + symbol.name + "`", data, decl.type_tok.pos)

    def visit_ExpressionNode(self, expr, parent_table):
        expr.set_type(self.expr_type_analyzer.visit(expr.node, parent_table))
        expr.set_constant(self.expr_is_constant(expr.node))

        if parent_table.scope_level == 0: # root scope
            if not expr.is_constant:
                raise_error("main.ess", "Expected constant expression at root scope", data, expr.tok_pos)

    def generic_visit(self, node, *args):
        pass

    ########################################################################### 
       
    def expr_is_constant(self, node):

        if isinstance(node, NumberNode):
            return True

        elif isinstance(node, BinOpNode):
            left_state = self.expr_is_constant(node.left)
            right_state = self.expr_is_constant(node.right)

            if left_state == False or right_state == False:
                return False
            return True

        elif isinstance(node, UnaryOpNode):
            return self.expr_is_constant(node.node)

        elif isinstance(node, VariableNode):
            return False

class ExpressionTypeAnalyzer(NodeVisitor):

    def visit_NumberNode(self, node, parent_table):
        
        if node.token.type == INTEGER:
            return TYPE_INT
        elif node.token.type == FLOAT:
            return TYPE_FLOAT
        else:
            raise Exception("SemanticAnalyzer.expr_type can not handle " + node.token.type + " NumberNodes")

    def visit_BinOpNode(self, node, parent_table):
        left_type = self.visit(node.left, parent_table)
        right_type = self.visit(node.right, parent_table)

        if left_type != right_type:
            raise_error("main.ess", "Cannot operate on `" + left_type.name + "` and `" + right_type.name + "`", data, node.op_tok.pos)

        return left_type

    def visit_UnaryOpNode(self, node, parent_table):
        return self.visit(node.node, parent_table)

    def visit_VariableNode(self, node, parent_table):

        var = parent_table.get_global(node.name_tok.value)
        if var == None:
            raise_error("main.ess", "Symbol `" + node.name_tok.value + "` not found", data, node.name_tok.pos)
        return var.type
        

###############################################################################
#                                                                             #
#  COMPILER                                                                   #
#                                                                             #
###############################################################################

#####################################
# Requirements                      #
#####################################

requirements = {
    "math": False
}

#####################################
# Expression Compiler               #
#####################################

class ExpressionCompiler(NodeVisitor):

    def compile(self, expression):
        self.result = ""
        self.visit(expression.node)
        return self.result

    def visit_NumberNode(self, node):
        self.result += "(" + node.token.value + ")"

    def visit_BinOpNode(self, node):
        
        if node.op_tok.value == "^":
            requirements["math"] = True
            self.result += "pow("
            self.visit(node.left)
            self.result += ","
            self.visit(node.right)
            self.result += ")"
        else:
            self.visit(node.left)
            self.result += node.op_tok.value
            self.visit(node.right)

    def visit_UnaryOpNode(self, node):
        self.result += node.op_tok.value
        self.visit(node.node)

    def visit_VariableNode(self, node):
        self.result += node.name_tok.value

#####################################
# Main Compiler                     #
#####################################

class Compiler(NodeVisitor):
    def __init__(self, program):
        self.expr_compiler = ExpressionCompiler()
        self.program = program
        self.result = ""

    def compile(self):
        for decl in self.program.decls:
            if isinstance(decl, VariableDeclarationNode):
                self.visit(decl, self.program.symbol_table)

        for assignment in self.program.assignments:
            self.visit(assignment, self.program.symbol_table)

        for decl in self.program.decls:
            if isinstance(decl, FunctionDeclarationNode):
                self.visit(decl)
        
        self.requirements()
        
        return self.result

    def requirements(self):

        if requirements["math"]:
            self.result = "#include <math.h>\n" + self.result

    ##########################################################################

    def visit_FunctionDeclarationNode(self, func):
        symbol = func.symbol_table.parent.get_global(func.name_tok.value)

        if symbol.type == TYPE_VOID:
            self.result += "void"
        elif symbol.type == TYPE_INT:
            self.result += "int"
        elif symbol.type == TYPE_FLOAT:
            self.result += "float"
        else:
            raise Exception("Compiler doesn't support built-in type " + func.type_tok.value + " as a function type")
        self.result += " " + symbol.name + "(){"

        for decl in func.decls:
            self.visit(decl, func.symbol_table)
            
        for statement in func.statements:
            self.visit(statement, func.symbol_table)
            
        self.result += "}"

    def visit_VariableDeclarationNode(self, var, parent_table):
        symbol = parent_table.get_global(var.name_tok.value)
        
        if symbol.type == TYPE_VOID:
            raise_error("main.ess", "Variables cannot be declared with type `void`", data, var.type_tok.pos)
        elif symbol.type == TYPE_INT:
            self.result += "int"
        elif symbol.type == TYPE_FLOAT:
            self.result += "float"
        else:
            raise Exception("Compiler doesn't support built-in type " + var.type_tok.value + " as a variable type")
        self.result += " " + symbol.name + ";"

    def visit_AssignmentNode(self, assignment, parent_table):
        symbol = parent_table.get_global(assignment.name_tok.value)
        
        self.result += symbol.name + "="
        self.visit(assignment.expr)
        self.result += ";"

    def visit_ExpressionNode(self, expression):
        self.result += self.expr_compiler.compile(expression)

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
    index2 = index
    while index2 >= len(string):
        index2 -= 1
    diff = index - index2 + 1
    line, col = get_line_col(string, index2)
    col += diff
    os.system("cls")
    print "An error has occured in `" + f + "`"
    print
    print "Error: " + error
    print "Ln: " + str(line) + "  Col: " + str(col)
    print
    print get_line(string, line-1)
    print " " * (col - 1) + "^"
    print
    os.system("pause")
    exit(0)

def get_symbol_type(symbol):
    if isinstance(symbol, VarSymbol):
        return "variable symbol"
    if isinstance(symbol, BuiltInTypeSymbol):
        return "built-in type"
    if isinstance(symbol, FunctionSymbol):
        return "function symbol"

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

    lexer = Lexer(data)
    parser = Parser(lexer)
    program = parser.parse()

    print json.dumps(program, default=lambda o: o.__dict__, indent=4, sort_keys=True)
    print

    SymbolTableBuilder().visit(program)
    SemanticAnalyzer().visit(program)
    compiler = Compiler(program)
    result = compiler.compile()

    dir_ = "project/build/c"

    if not os.path.exists(dir_):
        os.makedirs(dir_)

    f = open(dir_ + "/main.c", "w")
    f.write(result)
    f.close()

    os.system("pause")
