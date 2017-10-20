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
#  SYMBOLS                                                                    #
#                                                                             #
###############################################################################

class SymbolTables:
    def __init__(self, *args):
        self.tables = []
        self.length = 0
        for arg in args:
            self.add(arg)

    def add(self, table):
        self.tables.append(table)
        self.length += 1

    def pop(self):
        self.length -= 1
        return self.tables.pop()

    def any_has(self, name):
        return any(table.has(name) for table in self.tables)

    def get(self, name):
        for table in reversed(self.tables):
            symbol = table.get(name)
            if symbol == None:
                continue
            return symbol
        return None

class SymbolTable:
    def __init__(self):
        self.symbols = {}
    
    def add(self, symbol):
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

TYPE_VOID   = BuiltInTypeSymbol("void")
TYPE_INT    = BuiltInTypeSymbol("int")
TYPE_FLOAT  = BuiltInTypeSymbol("float")

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
    def visit(self, node, *args):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, *args)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

#####################################

class Program(AST):
    def __init__(self, decls, assignments):
        self.decls = decls
        self.assignments = assignments
        self.symbol_table = SymbolTable()
        self.symbol_table.add(TYPE_VOID)
        self.symbol_table.add(TYPE_INT)
        self.symbol_table.add(TYPE_FLOAT)

        symbol_tables = SymbolTables(self.symbol_table)
        
        for decl in decls:

            if self.symbol_table.has(decl.name_tok.value):
                raise_error("main.ess", "Symbol `" + decl.name_tok.value + "` has already been defined", data, decl.name_tok.pos)
            
            if isinstance(decl, FunctionDeclarationNode):
                self.symbol_table.add(FunctionSymbol(decl.name_tok.value, self.symbol_table.get(decl.type_tok.value)))
                decl.create_symbol_table(symbol_tables)
            
            elif isinstance(decl, VariableDeclarationNode):
                self.symbol_table.add(VarSymbol(decl.name_tok.value, self.symbol_table.get(decl.type_tok.value)))

class FunctionDeclarationNode(Node):
    def __init__(self, typ, name, args, decls, statements):
        self.type_tok = typ
        self.name_tok = name
        self.args = args
        self.decls = decls
        self.statements = statements

    def create_symbol_table(self, symbol_tables):

        self.symbol_table = SymbolTable()

        for decl in self.decls:

            if self.symbol_table.has(decl.name_tok.value):
                raise_error("main.ess", "Symbol `" + decl.name_tok.value + "` has already been defined", data, decl.name_tok.pos)
            
            if isinstance(decl, FunctionDeclarationNode):
                raise_error("main.ess", "Cannot define function in other function", data, decl.type_tok.pos)
                
            elif isinstance(decl, VariableDeclarationNode):
                self.symbol_table.add(VarSymbol(decl.name_tok.value, symbol_tables.get(decl.type_tok.value)))


class VariableDeclarationNode(Node):
    def __init__(self, typ, name):
        self.type_tok = typ
        self.name_tok = name
##        self.expr = expr
##
##        if expr == None:
##            if self.type_tok.value == 'int':
##                self.expr = ExpressionNode(NumberNode(Token(INTEGER, "0")))
##            elif self.type_tok.value == 'float':
##                self.expr = ExpressionNode(NumberNode(Token(FLOAT, "0.0")))
##            elif self.type_tok.value == "void":
##                raise_error("main.ess", "Variables cannot be defined with type `void`", data, self.type_tok.pos)

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

#####################################
# Parser                            #
#####################################

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
#  SEMANTIC ANALYZER                                                          #
#                                                                             #
###############################################################################

class SemanticAnalyzer(NodeVisitor):
    
    def visit_Program(self, program):
        tables = SymbolTables(program.symbol_table)
        
        for decl in program.decls:
            self.visit(decl, tables)

        for assignment in program.assignments:
            self.visit(assignment, tables)

    def visit_FunctionDeclarationNode(self, func, symbol_tables):
        self.visit_FunctionDeclarationNode_or_VariableDeclarationNode(func, symbol_tables)
        symbol_tables.add(func.symbol_table)

        for statement in func.statements:
            self.visit(statement, symbol_tables)
        
        symbol_tables.pop()

    def visit_VariableDeclarationNode(self, decl, symbol_tables):
        self.visit_FunctionDeclarationNode_or_VariableDeclarationNode(decl, symbol_tables)

    def visit_AssignmentNode(self, assignment, symbol_tables):
        self.visit(assignment.expr, symbol_tables)

        symbol = symbol_tables.get(assignment.name_tok.value)

        if symbol.type != assignment.expr.type:
            raise_error("main.ess", "Variable with type `" + symbol.type.name + "` cannot be assigned to type `" + assignment.expr.type.name + "`", data, assignment.expr.tok_pos)


    def visit_FunctionDeclarationNode_or_VariableDeclarationNode(self, decl, symbol_tables):
        symbol = symbol_tables.get(decl.type_tok.value)
        
        if symbol == None:
            raise_error("main.ess", "Symbol `" + decl.type_tok.value + "` not found", data, decl.type_tok.pos)
        
        if not isinstance(symbol, BuiltInTypeSymbol):
            raise_error("main.ess", "Expected built-in type but instead got " + get_symbol_type(symbol) + " `" + symbol.name + "`", data, decl.type_tok.pos)

    def visit_ExpressionNode(self, expr, symbol_tables):
        expr.set_type(self.expr_type(expr.node, symbol_tables))
        expr.set_constant(self.expr_is_constant(expr.node))

        if symbol_tables.length == 1: # root scope
            if not expr.is_constant:
                raise_error("main.ess", "Expected constant expression at root scope", data, expr.tok_pos)

        self.visit(expr.node, symbol_tables)

    def generic_visit(self, node, *args):
        pass

    ###########################################################################

    def expr_type(self, node, symbol_tables):
        
        if isinstance(node, NumberNode):
            if node.token.type == INTEGER:
                return TYPE_INT
            elif node.token.type == FLOAT:
                return TYPE_FLOAT
            else:
                raise Exception("SemanticAnalyzer.expr_type can not handle " + node.token.type + " NumberNodes")

        if isinstance(node, BinOpNode):
            left_type = self.expr_type(node.left, symbol_tables)
            right_type = self.expr_type(node.right, symbol_tables)

            if left_type != right_type:
                raise_error("main.ess", "Cannot operate on `" + left_type.name + "` and `" + right_type.name + "`", data, node.op_tok.pos)

            return left_type

        if isinstance(node, UnaryOpNode):
            return self.expr_type(node.node, symbol_tables)

        if isinstance(node, VariableNode):
            var = symbol_tables.get(node.name_tok.value)
            if var == None:
                raise_error("main.ess", "Symbol `" + node.name_tok.value + "` not found", data, node.name_tok.pos)
            return var.type

        raise Exception("SemanticAnalyzer.expr_type can not handle " + type(node).__name__)

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
# Compiler                          #
#####################################

class Compiler:
    def __init__(self, program):
        self.program = program
        self.result = ""

    def compile(self):
        symbol_tables = SymbolTables(self.program.symbol_table)
        self.variables(symbol_tables)
        self.functions(symbol_tables)
        self.requirements()
        return self.result
    
    ##########################################################################

    def requirements(self):

        if requirements["math"]:
            self.result = "#include <math.h>\n" + self.result

    ##########################################################################

    def functions(self, symbol_tables):
        for decl in self.program.decls:
            if isinstance(decl, FunctionDeclarationNode):
                symbol_tables.add(decl.symbol_table)
                self.function(decl, symbol_tables.get(decl.name_tok.value), symbol_tables)
                symbol_tables.pop()

    def variables(self, symbol_tables):
        for decl in self.program.decls:
            if isinstance(decl, VariableDeclarationNode):
                self.variable_decl(decl, symbol_tables.get(decl.name_tok.value))

        for assignment in self.program.assignments:
            self.variable_assignment(assignment, symbol_tables.get(assignment.name_tok.value))

    def function(self, func, symbol, symbol_tables):
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
            self.variable_decl(decl, symbol_tables.get(decl.name_tok.value))

        for statement in func.statements:
            if isinstance(statement, AssignmentNode):
                self.variable_assignment(statement, symbol_tables.get(statement.name_tok.value))

        self.result += "}"

    def variable_decl(self, var, symbol):
        if symbol.type == TYPE_VOID:
            raise_error("main.ess", "Variables cannot be declared with type `void`", data, var.type_tok.pos)
        elif symbol.type == TYPE_INT:
            self.result += "int"
        elif symbol.type == TYPE_FLOAT:
            self.result += "float"
        else:
            raise Exception("Compiler doesn't support built-in type " + var.type_tok.value + " as a variable type")
        self.result += " " + symbol.name + ";"

    def variable_assignment(self, var, symbol):
        self.result += symbol.name + "="
        self.result += self.expr(var.expr.node)
        self.result += ";"

    def expr(self, node):
        result = "("

        if isinstance(node, NumberNode):
            result += node.token.value
        
        elif isinstance(node, BinOpNode):
            if node.op_tok.value == "^":
                requirements["math"] = True
                result += "pow("
                result += self.expr(node.left)
                result += ","
                result += self.expr(node.right)
                result += ")"
            else:
                result += self.expr(node.left)
                result += node.op_tok.value
                result += self.expr(node.right)
        
        elif isinstance(node, UnaryOpNode):
            result += node.op_tok.value
            result += self.expr(node.node)

        elif isinstance(node, VariableNode):
            result += node.name_tok.value
        
        result += ")"
        return result
        

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
