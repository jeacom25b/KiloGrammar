import inspect
import os
import re


class Color:
    """
    colors defined as excape sequences
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


class TokenizerClass(Tokenizer):

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
             ["var", "[A-Za-z]+"],
             ["int", "-?[0-9]+"],
             ["float", "-?[0-9+]+\.[0-9]+"],
             ["whitespace", "[ \s]+"],
             ["(", "\("],
             [")", "\)"],
             ["+", "\+"],
             ["-", "\-"],
             ["*", "\*"],
             ["/", "\/"],
             ]

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


class ParserClass(Parser):

    @match({'whitespace'}, priority=0)
    def rule0_ignore_whitespace(parser, *matches):
        pop(parser, matches, 1)

    @match({'ADD', 'SUB'}, {'*', '/'}, {'ADD', 'SUB', 'MUL', 'DIV', 'int', 'float', 'var'}, priority=-1)
    def rule1_math_priority(parser, *matches):
        pop(parser, matches, 3)
        push(parser, matches, get_node(parser, matches, 0, 1), get_node(parser, matches, 0, 0), get_node(parser, matches, 0, 2), get_node(parser, matches, 1), get_node(parser, matches, 2))

    @match({'ADD', 'SUB', 'MUL', 'DIV', 'int', 'float', 'var'}, {'+', '-', '*', '/'}, {'ADD', 'SUB', 'MUL', 'DIV', 'int', 'float', 'var'}, priority=-2)
    def rule2_math(parser, *matches):
        pop(parser, matches, 3)
        push(parser, matches, node(parser, matches, pick_name(parser, matches, get_node(parser, matches, 1), ('+', '-', '*', '/'), ('ADD', 'SUB', 'MUL', 'DIV')), get_node(parser, matches, 1), get_node(parser, matches, 0), get_node(parser, matches, 2)))

    @match({'('}, {'ADD', 'SUB', 'MUL', 'DIV', 'int', 'float', 'var'}, {')'}, priority=-3)
    def rule3_parenthesis(parser, *matches):
        pop(parser, matches, 3)
        push(parser, matches, get_node(parser, matches, 1))



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


