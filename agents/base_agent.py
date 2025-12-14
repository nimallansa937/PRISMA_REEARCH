"""
Base Agent Classes for Multi-Agent Research Protocol.

All agents use ONLY DeepSeek and Gemini as LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMClient


class AgentTier(Enum):
    """Agent tier classification"""
    TIER1_EXECUTOR = 1      # Scripted, no LLM
    TIER2_SPECIALIST = 2    # Domain LLM (Gemini primary)
    TIER3_COUNCIL = 3       # Strategic LLM (DeepSeek primary)


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type,
            "payload": self.payload
        }


class BaseAgent(ABC):
    """
    Abstract base class for all research agents.
    
    Tier 1: No LLM (scripted execution)
    Tier 2: Gemini primary, DeepSeek fallback
    Tier 3: DeepSeek primary, Gemini fallback
    """
    
    def __init__(
        self,
        name: str,
        tier: AgentTier,
        description: str = ""
    ):
        self.name = name
        self.tier = tier
        self.description = description
        self.message_log: List[AgentMessage] = []
        
        # Initialize LLM client based on tier
        if tier == AgentTier.TIER1_EXECUTOR:
            self.llm = None  # No LLM for Tier 1
        elif tier == AgentTier.TIER2_SPECIALIST:
            # Gemini primary for Tier 2 (cheaper, good at analysis)
            self.llm = LLMClient(primary="gemini", fallback="deepseek")
        elif tier == AgentTier.TIER3_COUNCIL:
            # DeepSeek primary for Tier 3 (better reasoning)
            self.llm = LLMClient(primary="deepseek", fallback="gemini")
    
    @abstractmethod
    def execute(self, input_data: Dict) -> Dict:
        """Execute the agent's primary task"""
        pass
    
    def send_message(self, recipient: str, msg_type: str, payload: Dict) -> AgentMessage:
        """Send a message to another agent"""
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=msg_type,
            payload=payload
        )
        self.message_log.append(msg)
        return msg
    
    def receive_message(self, message: AgentMessage):
        """Receive and log a message"""
        self.message_log.append(message)
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            "name": self.name,
            "tier": self.tier.name,
            "description": self.description,
            "messages_processed": len(self.message_log),
            "llm_available": self.llm is not None
        }
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        temperature: float = 0.1
    ) -> Dict:
        """Call LLM with automatic fallback"""
        if self.llm is None:
            raise ValueError(f"Agent {self.name} is Tier 1 (no LLM)")
        
        return self.llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=schema,
            temperature=temperature
        )
    
    def __repr__(self):
        return f"<{self.name} ({self.tier.name})>"


class Tier1Agent(BaseAgent):
    """Base class for Tier 1 scripted agents (no LLM)"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, AgentTier.TIER1_EXECUTOR, description)


class Tier2Agent(BaseAgent):
    """Base class for Tier 2 specialist agents (Gemini primary)"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, AgentTier.TIER2_SPECIALIST, description)


class Tier3Agent(BaseAgent):
    """Base class for Tier 3 council agents (DeepSeek primary)"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, AgentTier.TIER3_COUNCIL, description)
