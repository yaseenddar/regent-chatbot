from abc import ABC, abstractmethod
from langchain.tools import BaseTool as LangChainBaseTool
class BaseTool(ABC):
    """Abstract base calss for tools that interact with the LLM."""

    def __init__(self,name:str,description:str):
        """
        Initializes a tool with a name and description.

        :param name: Name of the tool (converted to lowercase for consistency.
        :param description: A brief description of the tool.
        """

        if not isinstance(name,str):
            raise ValueError("Tool name must be a string.")
        
        self._name = name.lower()
        self._description = description
    
    @property
    def name(self):
        """Returns the name of the tool."""
        return self._name
    
    @property
    def description(self):
        """Returns the description of the tool."""
        return self._description
    
    @abstractmethod
    def __str__(self):
        """
        Abstract method that must be implemnted by all tools.

        :param query: The input query for the tools.
        :return: The output or response from the tool.

        """
        pass