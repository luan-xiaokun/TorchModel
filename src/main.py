"""TorchModel interpreter entrance"""
import argparse
from typing import Any, List, Tuple


from recipe_parser.torchmodel_parser import Lark_StandAlone, Transformer, Token
from training_recipe import recipe_builder


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


argument_parser = argparse.ArgumentParser(prog="TorchModel interpreter")
argument_parser.add_argument("filename")

args = argument_parser.parse_args()
parser = Lark_StandAlone(transformer=TorchModelConfigBuilder())

with open(args.filename, encoding="utf-8") as f:
    text = f.read()
parse_result = parser.parse(text)

recipe = recipe_builder(parse_result)
recipe.train()
