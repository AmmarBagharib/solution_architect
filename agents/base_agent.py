"""
Base Agent module that defines the core functionality for all agents in the system.
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    This class provides common functionality and interface requirements
    for all specialized agents.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the agent with a name and default settings.
        
        Args:
            name: The name of the agent. If not provided, a unique name will be generated.
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"{self.__class__.__name__}_{self.id[:8]}"
        self.logger = logging.getLogger(self.name)
        self.logger.info(f"Initializing agent {self.name}")
        self.state: Dict[str, Any] = {}
        
    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent's main functionality.
        
        Args:
            input_data: Data required by the agent to perform its task.
            
        Returns:
            Result of the agent's processing.
        """
        pass
    
    def update_state(self, data: Dict[str, Any]) -> None:
        """
        Update the agent's internal state.
        
        Args:
            data: New data to incorporate into the state.
        """
        self.state.update(data)
        self.logger.debug(f"Updated state: {list(data.keys())}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the agent's current state.
        
        Returns:
            The full internal state of the agent.
        """
        return self.state.copy()
    
    def log_info(self, message: str) -> None:
        """
        Log an information message.
        
        Args:
            message: The message to log.
        """
        self.logger.info(message)
        
    def log_error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: The error message to log.
        """
        self.logger.error(message)
        
    async def communicate(self, 
                        target_agent: 'BaseAgent', 
                        message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to another agent and get its response.
        
        Args:
            target_agent: The agent to communicate with.
            message: The message to send.
            
        Returns:
            The target agent's response.
        """
        self.logger.debug(f"Communicating with {target_agent.name}")
        response = await target_agent.run(message)
        return response
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that the input data contains all required keys.
        
        Args:
            input_data: The input data to validate.
            required_keys: A list of keys that must be present in the input data.
            
        Returns:
            True if the input is valid, False otherwise.
        """
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            self.log_error(f"Missing required input keys: {missing_keys}")
            return False
        return True