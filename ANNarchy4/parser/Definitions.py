## Lexicon
# OPERATORS
PLUS = '+'
MINUS = '-'
MULT = '*'
DIV = '/'
POW = '^'
IF = ['if', 'IF']
THEN = ['then', 'THEN']
ELSE = ['else', 'ELSE']
# EQUAL
EQUAL = '='
GREATER = ['>']
SMALLER = ['<']
# FUNCTIONS
COS = ['cos', 'Cos', 'COS', 'Cosinus', 'cosinus', 'COSINUS']
SIN = ['sin', 'Sin', 'SIN', 'Sinus', 'sinus', 'SINUS']
TAN = ['tan', 'Tan', 'TAN', 'Tangent', 'tangent', 'TANGENT']
ACOS = ['acos', 'arccos', 'ACOS', 'acosinus', 'ACOSINUS']
ASIN = ['asin', 'arcsin', 'ASIN', 'asinus', 'ASINUS']
ATAN = ['atan', 'arctan', 'ATAN', 'atangent', 'ATANGENT']
COSH = ['cosh', 'ch', 'COSH']
SINH = ['sinh', 'sh', 'SINH']
TANH = ['tanh', 'tanH', 'TANH']
ACOSH = ['acosh', 'ACOSH']
ASINH = ['asinh', 'ASINH']
ATANH = ['atanh', 'ATANH']
EXP = ['exp', 'Exp', 'EXP']
ABS = ['abs', 'fabs']
SQRT = ['sqrt', 'squaredroot']
LOG = ['log', 'ln']
LOG2 = ['log2']
POS = ['pos', 'positive']
NEG = ['neg', 'negative']
SUMS = ['sum']
# CONSTANTS
PI = ['pi', 'PI', 'Pi']
TRUE = ['True', 'true', 'TRUE']
FALSE = ['False', 'false', 'FALSE']
# BRACKETS
BRACKET_LEFT = '('
BRACKET_RIGHT = ')'
# VARIABLES
PRE_VAR = ['pre.']
POST_VAR = ['post.']

# GROUPS
OPERATORS = (PLUS, MINUS, MULT, DIV, POW)
FUNCTIONS = (COS, SIN, TAN, ACOS, ASIN, ATAN, COSH, SINH, TANH, ACOSH, ASINH, ATANH, EXP, ABS, SQRT, LOG, LOG2, POS, NEG)
TERNARY = (IF, THEN, ELSE)
CONSTANTS = (PI, TRUE, FALSE)
COMPARATORS = (GREATER, SMALLER)
BRACKETS = (BRACKET_LEFT, BRACKET_RIGHT)

# C++ EQUIVALENTS
cpp_equivalents = { 'cos': COS,
                    'sin': SIN,
                    'tan': TAN,
                    'acos': ACOS,
                    'asin': ASIN,
                    'atan': ATAN,
                    'cosh': COSH,
                    'sinh': SINH,
                    'tanh': TANH,
                    'acosh': ACOSH,
                    'asinh': ASINH,
                    'atanh': ATANH,
                    'exp': EXP,
                    'fabs': ABS,
                    'sqrt': SQRT,
                    'M_PI': PI,
                    'true': TRUE,
                    'false': FALSE,
                    'log': LOG,
                    'log2': LOG2,
                    'positive': POS,
                    'negative': NEG}

# LATEX EQUIVALENTS
latex_equivalents = { '\\cos': COS,
                    '\\sin': SIN,
                    '\\tan': TAN,
                    '\\text{acos}': ACOS,
                    '\\text{asin}': ASIN,
                    '\\text{atan}': ATAN,
                    '\\text{cosh}': COSH,
                    '\\text{sinh}': SINH,
                    '\\text{tanh}': TANH,
                    '\\text{acosh}': ACOSH,
                    '\\text{asinh}': ASINH,
                    '\\text{atanh}': ATANH,
                    '\\exp': EXP,
                    '\\text{abs}': ABS,
                    '\\sqrt': SQRT,
                    '\\pi': PI,
                    '\\log': LOG,
                    '\\log_2': LOG2}
                    
def cpp_equivalent(string):
    for key, val in cpp_equivalents.items():
        if string in val:
            return key
    print 'Error: the function', string, 'is not defined.'
                    
def latex_equivalent(string):
    for key, val in latex_equivalents.items():
        if string in val:
            return key
    return '\\text{'+string+ '}'

# DATA TYPE
DATA_TYPE = 'DATA_TYPE'

