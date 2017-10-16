###############################################################################
#                                                                             #
#  IMPORTS                                                                    #
#                                                                             #
###############################################################################

import os
import string

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

#####################################
# Tokens                            #
#####################################

##INTEGER    =    'INTEGER'
##PLUS       =    'PLUS'
##MINUS      =    'MINUS'
##MUL        =    'MUL'
##DIV        =    'DIV'
##CARET      =    'CARET'
##LPAREN     =    'LPAREN'
##RPAREN     =    'RPAREN'
##EOF        =    'EOF'

IDENTIFIER  =   'IDENTIFIER'
KEYWORD     =   'KEYWORD'
LPAREN      =   'LPAREN'
RPAREN      =   'RPAREN'
LBRACE      =   'LBRACE'
RBRACE      =   'RBRACE'
SEMI        =   'SEMI'
EOF         =   'EOF'

lexer_pos = 0

class Token(object):
    def __init__(self, typ, val):
        self.type = typ
        self.value = val
        self.lexer_pos = lexer_pos

#####################################
# Lexer                             #
#####################################

class Lexer(object):
    def __init__(self, text):
        global lexer_pos
        self.text = text
        self.pos = 0
        lexer_pos = self.pos
        self.current_char = self.text[self.pos]

    def error(self):
        raise_error("main.ess", "Illegal character", self.text, self.pos)

    def advance(self):
        global lexer_pos
        self.pos += 1
        lexer_pos = self.pos
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

##    def integer(self):
##        result = ''
##        while self.current_char is not None and self.current_char.isdigit():
##            result += self.current_char
##            self.advance()
##        return int(result)

    def identifier(self):

        letters = list(string.ascii_uppercase + string.ascii_lowercase)
        digits = list(range(10))

        result = ""

        while self.current_char in letters + digits:
            result += self.current_char
            self.advance()

        return result
            

    def get_next_token(self):
        
        while self.current_char is not None:

            letters = list(string.ascii_uppercase + string.ascii_lowercase)

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == '{':
                self.advance()
                return Token(LBRACE, '{')

            if self.current_char == '}':
                self.advance()
                return Token(RBRACE, '}')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char in letters:
                return Token(IDENTIFIER, self.identifier())

##            if self.current_char.isdigit():
##                return Token(INTEGER, self.integer())
##
##            if self.current_char == '+':
##                self.advance()
##                return Token(PLUS, '+')
##
##            if self.current_char == '-':
##                self.advance()
##                return Token(MINUS, '-')
##
##            if self.current_char == '*':
##                self.advance()
##                return Token(MUL, '*')
##
##            if self.current_char == '/':
##                self.advance()
##                return Token(DIV, '/')
##
##            if self.current_char == '^':
##                self.advance()
##                return Token(CARET, '^')
##
##            if self.current_char == '(':
##                self.advance()
##                return Token(LPAREN, '(')
##
##            if self.current_char == ')':
##                self.advance()
##                return Token(RPAREN, ')')

            self.error()

        return Token(EOF, None)

###############################################################################
#                                                                             #
#  SYMBOLS                                                                    #
#                                                                             #
###############################################################################

class SymbolTable:
    def __init__(self):
        self.symbols = {}
    
    def add(self, symbol):
        if symbol.name in self.symbols:
            raise_simple_error("main.ess", "Symbol `" + symbol.name + "` has already been defined")
        self.symbols[symbol.name] = symbol

    def has(self, name):
        return name in self.symbols

    def get(self, name):
        return self.symbols.get(name, None)

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

TYPE_VOID = BuiltInTypeSymbol("void")

###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST(object):
    pass

class Node(AST):
    pass

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

#####################################

class Program(AST):
    def __init__(self, decls):
        self.decls = decls
        self.symbol_table = SymbolTable()
        self.symbol_table.add(TYPE_VOID)
        
        for decl in decls:
            if isinstance(decl, FunctionDeclarationNode):
                self.symbol_table.add(FunctionSymbol(decl.name, decl.type))

class FunctionDeclarationNode(Node):
    def __init__(self, typ, name, args, statements):
        self.type = typ
        self.name = name
        self.args = args
        self.statements = statements

class FunctionCallNode(Node):
    def __init__(self, name, params):
        self.name = name
        self.params = params

##class NumberNode(Node):
##    def __init__(self, token):
##        self.token = token
##        self.value = token.value
##
##class BinOpNode(Node):
##    def __init__(self, left, op, right):
##        self.left = left
##        self.token = self.op = op
##        self.right = right
##
##class UnaryOpNode(Node):
##    def __init__(self, op, expr):
##        self.token = self.op = op
##        self.expr = expr

#####################################
# Parser                            #
#####################################

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise_error("main.ess", "Invalid syntax", self.lexer.text, self.current_token.lexer_pos)

    def tryeat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
            return True
        return False

    def eat(self, *token_types):
        value = self.current_token.value
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
        decls = self.declarations()
        return Program(decls)

    def declarations(self):
        decls = []
        while self.token_is(IDENTIFIER):
            decls.append(self.declaration())
        return decls

    def declaration(self):
        typ = self.eat(IDENTIFIER)
        name = self.eat(IDENTIFIER)
        args = self.function_definition_argument_list()
        statements = self.compound_statement()
        return FunctionDeclarationNode(typ, name, args, statements)

    def function_definition_argument_list(self):
        self.eat(LPAREN)
        self.eat(RPAREN)
        return []

    def compound_statement(self):
        statements = []
        self.eat(LBRACE)
        while not self.token_is(RBRACE) and not self.token_is(EOF):
            statements.append(self.statement())
        self.eat(RBRACE)
        return []

    def statement(self):
        return

    def argument_list(self):
        self.eat(LPAREN)
        self.eat(RPAREN)
        return []

###############################################################################
#                                                                             #
#  SEMANTIC ANALYZER                                                          #
#                                                                             #
###############################################################################

class SemanticAnalyzer(NodeVisitor):
    def __init__(self):
        self.symtab = SymbolTable()

    def visit_Program(self, program):
        for decl in program.decls:
            if not program.symbol_table.has(decl.type):
                raise_simple_error("main.ess", "Symbol " + decl.type + " was not declared")

    def generic_visit(self, node):
        pass

###############################################################################
#                                                                             #
#  COMPILER                                                                   #
#                                                                             #
###############################################################################

class Compiler:
    def __init__(self, program):
        self.program = program
        self.result = ""

    def compile(self):
        self.compile_functions()
        return self.result

    def compile_functions(self):
        for decl in self.program.decls:
            if isinstance(decl, FunctionDeclarationNode):
                self.compile_function(decl)

    def compile_function(self, func):
        result = ""
        if func.type == "void":
            result += "void"
        result += " " + func.name + "(){}"
        self.result += result + "\n"
        

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
    #os.system("cls")
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
    #os.system("cls")
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

    import json
    print json.dumps(program, default=lambda o: o.__dict__, indent=4, sort_keys=True)
    print

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
