"""
This script is a parser for the kilogrammar language
It compiles it into a parser by just reusing its own code


To use it just type the command:
python kilogrammar.py my_parser.kg -compile > output_parser.py


to watch the parser working in interactive mode you can do:
python kilogrammar.py my_parser.kg -interactive -color



This file is quite long, to find meaningful sections, just search for
names in this table of contents:

    colors used for printing:
        class Color

    functions used to make the interactive visualization:
        def pretty_lines
        def pretty_print

    main Node class used by the parser:
        class Node

    main Token class used by the parser:
        class Token

    class that implement tokenizing code:
        class Tokenizer

    class that implements parsing code as a stack machine:
        class Parser

    code used to simplify the implementation of new parser rules using decorators:
        def match
        class MatchRuleWrapper

    default functions avaliable to the kilogrammar language:
        KG_BUILTINS = "

    main tokenizer loop:
        while self.char_ptr < len(self.text):

    main parser loop:
        for name, rule in rules

    Kilogrammar language parser class:
        class KiloParser(Parser):

    Kilogrammar tokenizer class:
        class KiloTokenizer(Tokenizer):


"""






# this tag is used to mark the start of the section that
# is going to be copied as a new file

# TAG1: reutilize code start
import inspect
import os
import re


class Color:
    """
    colors defined as escape sequences for fancy output.
    easy to use but dont work on all terminals
    https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    @classmethod
    def enable(cls):
        cls.red = "\u001b[31m"
        cls.yellow = "\u001b[38;5;221m"
        cls.pink = "\u001b[38;5;213m"
        cls.cyan = "\u001b[38;5;38m"
        cls.green = "\u001b[38;5;112m"
        cls.reset = "\u001b[0m"

    @classmethod
    def disable(cls):
        cls.red = ""
        cls.yellow = ""
        cls.pink = ""
        cls.cyan = ""
        cls.green = ""
        cls.reset = ""

Color.disable()

class Node:
    """
    Main class to construct an abstract syntax tree,
    It is expected to encapsulate instances of Node() or Token()
    """
    def __init__(self, node_type, contents):
        self.type = node_type
        assert(type(contents) == list)
        self.contents = contents

    def pretty_lines(self, out_lines=None, indent_level=0):
        """
        return pretyly formated lines
        containing the contents of its nodes and subnodes
        in a printable and human-readable form
        """
        if out_lines is None:
            out_lines = []

        out_lines.append("".join(("    " * indent_level, repr(self), ":")))
        for sub_thing in self.contents:
            if isinstance(sub_thing, Node):
                sub_thing.pretty_lines(out_lines, indent_level + 1)
            else:
                out_lines.append("".join(
                    ("    " * (indent_level + 1), repr(sub_thing))))
        return out_lines

    def __hash__(self): # used for easy comparation with sets
        return hash(self.type)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.type == other.type
        else:
            return self.type == other

    def __repr__(self):
        return f"{Color.green}{self.type}{Color.reset}"


def panic(msg, line, col, text):
    """
    raise an SyntaxError and display the position of the error in the text
    """
    text_line = text.split("\n")[line]
    arrow = " " * col + "^"
    raise SyntaxError("\n".join(("",
                                 text_line,
                                 arrow,
                                 msg,
                                 f" line: {line + 1}, collumn: {col + 1}")))

class Token:
    """
    Main token class,
    its supposed to encapsulate snippets of the text being parsed
    """
    def __init__(self, token_type, contents, line=None, col=None):
        self.type = token_type
        self.contents = contents
        self.line = line
        self.col = col

    def __hash__(self):
        return hash(self.type)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.type == other.type
        else:
            return self.type == other

    def __repr__(self):
        if self.contents in {None, ""}:
            return f"{Color.cyan}{self.type} {Color.reset}"
        return f"{Color.cyan}{self.type} {Color.pink}{repr(self.contents)}{Color.reset}"


class Tokenizer:
    """
    Main Tokenizer class, it parses the text and makes a
    list of tokens matched based on the rules defined for it

    it starts trying to match rules at the start of the file
    and when the first rule matches, it saves the match as a token and
    move forward by the length of the match to the next part of the text
    unless the rule defines a callback function, in this case the callback
    has to move to the next token using the feed() method

    If no rule matches any part of the text, it calls a panic()
    """
    rules = ["text", r"(?:\n|.)*"]

    def __init__(self, text, skip_error=False):
        self.text = text
        self.tokens = []
        self.errors = []
        self.skip_error = skip_error
        self.char_ptr = 0
        self.line_num = 0
        self.col_num = 0
        self.preprocess()
        self.tokenize()

    def preprocess(self):
        pass

    @staticmethod
    def default_callback(self, match, name):
        self.push_token(name, match[0])
        self.feed(len(match[0]))
        return len(match[0]) > 0

    def push_token(self, type, value=None):
        self.tokens.append(Token(type, value, self.line_num, self.col_num))

    def pop_token(self, index=-1):
        return self.tokens.pop(-1)

    def feed(self, n):
        for _ in range(n):
            if self.text[self.char_ptr] == "\n":
                self.line_num += 1
                self.col_num = -1
            self.char_ptr += 1
            self.col_num += 1

    def tokenize(self):
        import re
        import inspect
        rules = []

        self.preprocess()

        for rule in self.rules:
            if len(rule) == 2:
                (name, regex), callback = rule, self.default_callback
            elif len(rule) == 3:
                name, regex, callback = rule
            else:
                raise TypeError(f"Rule not valid: {rule}")

            try:
                regex = re.compile(regex)

            except Exception as e:
                print(str(e))
                raise TypeError(f"{type(self)}\n {name}: {repr(regex)}\n"
                                f"regex compilation failed")

            rules.append((name, regex, callback))

        while self.char_ptr < len(self.text):

            for name, regex, callback in rules:
                match = regex.match(self.text, self.char_ptr)
                if match:
                    done = callback(self, match, name)
                    if done:
                        break

            else:
                err = (f"Unexpected character: {repr(self.text[self.char_ptr])}",
                       self.line_num, self.col_num)

                if self.skip_error:
                    self.errors.append(err)
                    self.feed(1)
                else:
                    panic(*err, self.text)


class MatchRuleWrapper:
    """
    Encapsulates a parser rule definition and tests it against the
    parser stack, if a match is found, it calls its
    callback function that does its thing on the parser stack.
    """
    def __init__(self, func, rules, priority=0):
        self.func = func
        self.rules = rules
        self.priority = priority

    def __call__(self, parser, *args):
        n = len(parser.stack)
        if len(self.rules) > n or len(self.rules) > n:
            return

        i = 0
        for rule in reversed(self.rules):
            item = parser.stack[-1 - i]
            i += 1
            if not (rule is None or item.type in rule):
                break
        else:
            matches = parser.stack[-len(self.rules):]
            self.func(parser, *matches)
            return matches


def match(*args, priority=0):
    """
    returns decorator that helps defining parser rules and callbacks as
    if they were simple instance methods.
    In reality those methods are turned into MatchRuleWrapper callbacks
    """
    import inspect
    for arg in args:
        if not isinstance(arg, (set, str)) and arg is not None:
            raise TypeError(f"match_fun() invalid argument: {arg}")

    match_rules = []
    for arg in args:
        if arg == None or type(arg) == set:
            match_rules.append(arg)
        elif isinstance(arg, str):
            match_rules.append({s for s in arg.split("|") if s})
        else:
            raise TypeError(f"wrong type of argumment: {type(arg)}, {arg}")

    arg_count = len([type(arg) for arg in args]) + 1

    def decorator(func):
        paramaters = inspect.signature(func).parameters
        if len(paramaters) is not arg_count:
            if not inspect.Parameter.VAR_POSITIONAL in {p.kind for p in paramaters.values()}:
                raise TypeError(
                    f"function {func} does not contain {arg_count} argumments")
        return MatchRuleWrapper(func, match_rules, priority)

    return decorator


class Parser:
    """
    A stack machine that simply run its rules on its stack,
    every time no rule maches the contents of the stack
    a new token is pushed from the token list
    """
    def __init__(self, tokenizer, preview=999999, token_preview=5):
        self.tokens = tokenizer.tokens
        self.text = tokenizer.text
        self.token_ptr = 0
        self.stack = []
        self.preview = preview
        self.token_preview=token_preview
        self.parse()

    def push(self, node_type, contents=None):
        if contents is None:
            contents = []
        if type(node_type) in {Node, Token}:
            self.stack.append(node_type)
        else:
            self.stack.append(Node(node_type, contents))

    def pop(self, repeat=1, index=-1):
        for _ in range(repeat):
            self.stack.pop(index)

    def parse(self):
        rules = [(name, rule) for name, rule in inspect.getmembers(self)
                 if isinstance(rule, MatchRuleWrapper)]

        rules = sorted(rules, key=lambda r: -r[1].priority)

        while True:
            for name, rule in rules:
                matched = rule(self)
                if matched:
                    break

            else:
                if not self.token_ptr < len(self.tokens):
                    break
                self.stack.append(self.tokens[self.token_ptr])
                self.token_ptr += 1

            if self.preview > 0:
                self.pretty_print(self.preview, self.token_preview)

                print("stack:", self.stack, "\n")

                if matched:
                    print("matched rule:", name, matched, "\n")
                else:
                    print("no rule matched\n")
                inp = input("  Hit enter to continue, type 'e' to exit: ")
                if inp == "e":
                    self.preview = 0
                os.system("cls" if os.name == "nt" else "clear")

    def pretty_print(self, maximun_tree, maximun_tokens):
        lines = []
        for thing in self.stack:
            if isinstance(thing, Node):
                lines.extend(thing.pretty_lines())
            else:
                lines.append(repr(thing))

        display_lines = lines[max(-maximun_tree, -len(lines)):]
        if len(display_lines) < len(lines):
            print("...")

        print("\n".join(display_lines))
        print("\nNext tokens:")

        for i, token in enumerate(self.tokens[self.token_ptr:]):
            print(token)
            if i == maximun_tokens:
                break


class KiloTokenizer(Tokenizer):

    last_indent = None
    indent_deltas = []

    def handle_indent(self, match, name):
        n = len(match[1])
        if self.last_indent is None:
            self.last_indent = n

        delta = n - self.last_indent

        self.last_indent = n

        if delta > 0:
            self.push_token("indent_increase")
            self.indent_deltas.append(delta)

        elif delta < 0:
            while delta < 0 and self.indent_deltas:
                self.push_token("indent_decrease")
                delta += self.indent_deltas.pop(-1)
            if delta > 0:
                self.push_token("inconsistent_indent")

    rules = [["indent", r"\n([ \t]*)(?:[^ \t\n])", handle_indent],
# TAG1: reutilize code end
             ["newline", r"\n"],
             ["string", r""""(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'"""],
             ["whitespace", r"[ ;\t]+"],
             ["comment", r"#.*\n"],
             ["integer", r"(-?[0-9]+)\b"],
             ["rule", r"rule"],
             ["case", r"case"],
             ["keyword", r"keyword"],
             ["token", r"token"],
             ["word", r"\b[A-Za-z_]+[A-Za-z0-9_]*\b"],
             ["name", r"[^0-9\[\]\(\);\| \t\'\"\n,#>]+"],
             ["pipe", r"\|"],
             ["(", r"\("],
             [")", r"\)"],
             ["[", r"\["],
             ["]", r"\]"],
             [",", r","]]

    def preprocess(self):
        shorthand_re = re.compile(
            r"""shorthand\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')"""
        )
        new_lines = []
        shorthands = []
        for line in self.text.split("\n"):

            for replace, to in shorthands:
                replaced_line = line.replace(replace, to)
                if replaced_line != line:
                    line = replaced_line

            match = shorthand_re.match(line)
            if match:
                shorthands.append((match[1][1:-1], match[2][1:-1]))
                new_lines.append("".join(("# processed ", line)))
            else:
                new_lines.append(line)
        self.text = "\n".join(new_lines)


class KiloParser(Parser):
    """
    Implementation of the Kilogrammar parser
    """
    # =================================================
    # === tokens and keywords
    # =================================================

    @match("whitespace|newline|comment")
    def ignore(self, ig):
        self.pop(1)

    @match("token", "name|word", priority=1)
    def token_start(self, token, name):
        self.pop(2)
        self.push("TOKEN_DEF_START", [name])

    @match("TOKEN_DEF_START", "string", priority=1)
    def token_def_stage_1(self, tdef, string):
        self.pop(1)
        tdef.contents.append(string)
        tdef.type = "TOKEN_DEFINITION"

    @match("keyword", "string", priority=1)
    def keyword_def(self, keyword, string):
        self.pop(2)
        self.push("KEYWORD", [string])

    # =================================================
    # === rule definitions
    # =================================================

    @match("rule", "word")
    def rule_name(self, rule, name):
        self.pop(2)
        self.push("RULE_DEF_NAME", [name])

    @match("RULE_DEF_NAME", "(")
    def rule_def_start(self, name, par):
        self.pop(1)
        name.contents.append(Node("MATCH_LIST", []))
        name.type = "RULE_DEF_MATCH_LIST"

    @match("RULE_DEF_MATCH_LIST", "NAME_GRP", ",|)")
    def rule_def_extend(self, rule, names, sep):
        self.pop(2)
        rule.contents[1].contents.append(names)
        self.push(sep)

    @match("RULE_DEF_MATCH_LIST", ",")
    def rule_strip_comma(self, rule, comma):
        self.pop(1)

    @match("RULE_DEF_MATCH_LIST", ")")
    def rule_natch_list_finish(self, rule, par):
        rule.type = "RULE_DEF"
        self.pop(1)

    @match("RULE_DEF", "BLOCK")
    def rule_finish(self, rule, block):
        self.pop(1)
        rule.contents.append(block)
        rule.type = "RULE_DEFINITION"

    # =================================================
    # === name groups
    # =================================================

    @match("NAME_GRP", "pipe", "name|word|string", priority=2)
    def mane_grp_extend(self, grp, pipe, name):
        self.pop(2)
        grp.contents.append(name)

    @match("name|word|string", ",|pipe|)", priority=-1)
    def name_grp(self, name, sep):
        self.pop(2)
        self.push("NAME_GRP", [name])
        self.push(sep)

    # =================================================
    # === indent blocks
    # =================================================

    @match("FUNC_CALL", "indent_decrease")
    def block_end(self, call, indent):
        self.pop(2)
        self.push("BLOCK_END", [call])

    @match("FUNC_CALL", "BLOCK_END", priority=0)
    def block_end_expand(self, call, block):
        self.pop(2)
        block.contents.append(call)
        self.push(block)

    @match("indent_increase", "BLOCK_END", priority=1)
    def block_finish(self, indent, block):
        self.pop(2)
        self.push("BLOCK", list(reversed(block.contents)))

    # =================================================
    # === function calls
    # =================================================

    @match("word", "(")
    def func_call_start(self, name, p):
        self.pop(2)
        self.push("FUNC_CALL_START", [name, Node("ARGS", [])])

    @match("FUNC_CALL_START", "indent_increase|indent_decrease|inconsistent_indent")
    def ignore_indent(self, call, indent):
        self.pop(1)

    @match("FUNC_CALL_START", ",")
    def ignore_comma(self, func, separator):
        self.pop(1)

    @match("FUNC_CALL_START", "integer|NAME_GRP|FUNC_CALL", ",|)")
    def add_func_arg(self, func, arg, separator):
        self.pop(2)
        func.contents[1].contents.append(arg)
        self.push(separator)

    @match("FUNC_CALL_START", ")")
    def func_call_finish(self, func, par):
        self.pop(1)
        func.type = "FUNC_CALL"

    # =================================================
    # === Node Indexing
    # =================================================

    @match("INDEXES", ",|)")
    def indexes_to_func(self, indexes, sep):
        self.pop(2)
        self.push("FUNC_CALL", [Token("name", "get_node"),
                                Node("ARGS", indexes.contents)])
        self.push(sep)

    @match("[", "integer", "]")
    def make_index(self, sq, n, sq1):
        self.pop(3)
        self.push("INDEXES", [n])

    @match("INDEXES", "INDEXES")
    def sub_index(self, i, j):
        self.pop(1)
        i.contents.append(j.contents[0])


KG_BUILTINS = """
#  ============================================
#        Kilogrammar language builtins
#  ============================================

def push(parser, matches, *args):
    for arg in args:
        parser.stack.append(arg)

def pop(parser, matches, *args):
    if len(args) == 0:
        parser.stack.pop(-1)
    else:
        for _ in range(args[0]):
            parser.stack.pop(-1)

def node(parser, matches, name_grp, *args):
    return Node(name_grp[0], list(args))

def pick_name(parser, matches, name_selector, name_grp_from, name_grp_to):
    if isinstance(name_selector, (Node, Token)):
        name_selector = name_selector.type

    elif isinstance(name_selector, tuple):
        name_selector = name_selector[0]

    return (name_grp_to[name_grp_from.index(name_selector)],)

def get_node(parser, matches, *args):
    node = matches[args[0]]
    for index in args[1:]:
        node = node.contents[index]
    return node

"""

KG_BUILTINS_FUNC_LIST = [
    "push",
    "pop",
    "node",
    "pick_name",
    "get_node"
]


MAIN = r"""

if __name__ == '__main__':
    import sys

    if "-color" in sys.argv:
        Color.enable()

    text = None
    if "-type" in sys.argv:
        text = input("\n\n   input >>>")

    elif len(sys.argv) > 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            text = f.read()

    if text is not None:
        if '-interactive' in sys.argv:
            preview_length = 999999
        else:
            preview_length = 0
        tokens = TokenizerClass(text)
        parser = ParserClass(tokens, preview=preview_length)

        parser.pretty_print(999999, 999999)

    else:
        print("this script seems to not have syntax errors.")

"""

def validate(parser):
    for node in parser.stack:
        if  node.type not in\
                {"indent_decrease",
                 "indent_increase",
                 "RULE_DEF",
                 "TOKEN_DEFINITION",
                 "KEYWORD",
                 "RULE_DEFINITION"}:

            while isinstance(node, Node): #find a leaf token
                node = node.contents[0]
            panic(f"untexpected token: {node}", node.line, node.col, parser.text)

def parser_compile(parser):

    rule_defs = []
    token_defs = []
    keyword_defs = []

    def extract_high_level_parts(contents):
        for node in contents:
            if isinstance(node, Node):
                extract_high_level_parts(node.contents)
            if node == "RULE_DEFINITION":
                rule_defs.append(node)
            elif node == "TOKEN_DEFINITION":
                token_defs.append(node)
            elif node == "KEYWORD":
                keyword_defs.append(node)

    extract_high_level_parts(parser.stack)

    final_lines = []

    # recicling a usefull piece of code that cant be expressed directly using
    # this language,
    with open(__file__, "r") as myself:
        myself.seek(0)
        lines = myself.readlines()

        start = 0
        end = 0
        for i, line in enumerate(lines):
            if line.startswith("# TAG1: reutilize code start"):
                start = i
            elif line.startswith("# TAG1: reutilize code end"):
                end = i

        code = "".join(lines[1 + start:end])[0:-1]

        code = code.replace("KiloTokenizer", "TokenizerClass")
        final_lines.append(code)

    for token_def in token_defs:
        name = token_def.contents[0].contents
        regex = token_def.contents[1].contents[1:-1]
        final_lines.append(f'             ["{name}", "{regex}"],')

    for keyword_def in keyword_defs:
        name = keyword_def.contents[0].contents[1:-1]
        regex = re.escape(name)
        final_lines.append(f'             ["{name}", "{regex}"],')
    final_lines.append("             ]")

    def make_match_list(match_list):
        args = []
        for name_grp in match_list.contents:
            arg = []
            for name in name_grp.contents:
                if name == "string":
                    arg.append(f"'{name.contents[1:-1]}'")
                elif name in {"word", "name"}:
                    arg.append(f"'{name.contents}'")
            args.append("".join(("{", ", ".join(arg), "}")))
        return ", ".join(args)

    def make_func_call(node):
        contents = node.contents
        name_token = contents[0]
        name = name_token.contents

        if name not in KG_BUILTINS_FUNC_LIST:
            panic(f"function does not exist: {name}",
                    line=name_token.line,
                    col=name_token.col,
                    text=parser.text)

        argumments = contents[1].contents
        args = []

        for arg in argumments:
            if arg.type in "integer":
                args.append(arg.contents)
            elif arg.type == "FUNC_CALL":
                args.append(make_func_call(arg))
            elif arg.type == "NAME_GRP":
                args.append(repr(tuple(node.contents for node in arg.contents)))

        return f"{name}(parser, matches, {', '.join(args)})"

    final_lines.extend([KG_BUILTINS])
    final_lines.append("class ParserClass(Parser):")
    final_lines.append("")

    for i, rule in enumerate(rule_defs):
        rule = rule.contents
        func_name = rule[0].contents
        block = rule[2].contents
        match_args = make_match_list(rule[1])
        final_lines.append(f"    @match({match_args}, priority={-i})")
        final_lines.append(f"    def rule{i}_{func_name}(parser, *matches):")
        for func_call in block:
            final_lines.append(f"        {make_func_call(func_call)}")

        final_lines.append("")

    final_lines.append(MAIN)

    for line in final_lines:
        print(line)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            tok = KiloTokenizer(f.read() + "\n;")

        if "-color" in sys.argv:
            Color.enable()

        if "-compile" in sys.argv:
            parser = KiloParser(tok, preview=0)
            validate(parser)
            parser_compile(parser)

        else:
            if "-interactive" in sys.argv:
                preview = 999999
            else:
                preview = 0

            parser = KiloParser(tok, preview=preview)
            parser.pretty_print(999999, 999999)
            validate(parser)
