import math
import numexpr
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Python-style mathematical expression, e.g. '2**5' or 'pi*2**2'"
    )


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Evaluate a Python-style mathematical expression.
    Supports constants: pi and e.

    Examples:
    - 2+2
    - 532432 * 5
    - 2**4
    - pi * 2**2
    - 42343**(1/5)
    """
    if not expression or not expression.strip():
        return "Expression cannot be empty."

    try:
        result = numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict={"pi": math.pi, "e": math.e},
        )
        return str(result.item()) if hasattr(result, "item") else str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


if __name__ == "__main__":
    tools = [calculator]

    print(calculator.invoke("2+3"))
    print(calculator.invoke("pi * 2**2"))