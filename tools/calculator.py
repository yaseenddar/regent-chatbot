import math

from pydantic import BaseModel
import numexpr
from tools.base_tool import BaseTool

class Calculator(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description=(
            "A tool that can perform mathematical calculations. "
            "before calling convert natural langguage to proper Pyhton-stype math exxpresssions."
            "Example: '2+2','532432 * 5','2**4','pi * 2**2', 42343**(1/5)'."
            "Supports constants like pi and e."
        ))
        self.local_dict={"pi":math.pi,"e":math.e}

    def run(self,query:str) -> str:
        """
        Evaluate a mathematical expression provided numexpr
        """

        if not query or not query.strip():
            return "Expression cannot be empty."
        
        try:
            result = numexpr.evaluate(
                query.strip(),
                global_dict={},
                local_dict = self.local_dict
            )
            return str(result.item()) if hasattr(result,"item") else str(result)
        except Exception as e:
            return f"Error Evaluating expresssion: {str(e)}"

    def __str__(self):
        return f"Tool Name: {self.name}\nDescription: {self.description}"
# fir standalone testing

if __name__ == "__main__":
    calculator = Calculator()
    test_expressions = [
        '2+3',
        "32534 * 4",
        '5435 + (4*3)',
        'e**3'
    ]

    for exp in test_expressions:
        print(f"Expression: {exp} => Result: {calculator.run(exp)}")