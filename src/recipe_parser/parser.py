"""Parser for the torch-model language"""
from typing import Any, List, Tuple

from lark import Lark, Token
from lark.visitors import Transformer


class TorchModelConfigBuilder(Transformer):
    """Transform the AST of the torch-model DSL into a Python object"""

    def int(self, number: List[Token]) -> int:
        """Transformation rule for int"""
        (number,) = number
        return int(number)

    def float(self, number: List[Token]) -> float:
        """Transformation rule for float"""
        (number,) = number
        return float(number)

    def string(self, string: List[Token]) -> str:
        """Transformation rule for string"""
        (string,) = string
        return str(string[1:-1])

    def long_string(self, string: List[Token]) -> str:
        """Transformation rule for long string"""
        (string,) = string
        return str(string[3:-3])

    def const_true(self, _: List[Token]) -> bool:
        """Transformation rule for boolean True"""
        return True

    def const_false(self, _: List[Token]) -> bool:
        """Transformation rule for boolean False"""
        return False

    def key(self, key: List[Token]) -> str:
        """Transformation rule for key"""
        (key,) = key
        return str(key)

    def binding(self, pair: List[Any]) -> Tuple[str, Any]:
        """Transformation rule for binding"""
        return tuple(pair)

    def entry_name(self, name: List[Token]) -> str:
        """Transformation rule for entry_name"""
        (name,) = name
        return str(name)

    def entry_def(self, definition: List[Any]) -> Any:
        """Transformation rule for entry_def"""
        entry_name, method, *bindings = definition
        entry = {
            entry_name: {
                "class": str(method)[1:-1],
            }
        }
        entry[entry_name].update(bindings)
        return entry

    def start(self, entries: List[dict]) -> dict:
        """Transformation rule for recipe"""
        result = {}
        for entry in entries:
            result.update(entry)
        return result


with open("src/recipe_parser/torchmodel.lark", encoding="utf-8") as f:
    grammar = f.read()
parser = Lark(grammar, parser="lalr", transformer=TorchModelConfigBuilder())


if __name__ == "__main__":
    with open("examples/example.tm", encoding="utf-8") as f:
        text = f.read()
    parse_result = parser.parse(text)
    print(parse_result.pretty())
