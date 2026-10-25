"""Conversation state management for multi-turn support sessions.

Tracks:
- Turn sequence and timestamps
- Current customer ID and order ID
- Caller verification status
- Extracted entities and pending clarification needs
- Coreference resolution across follow-ups
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ConversationTurn:
    turn_index: int
    user_message: str
    assistant_reply: str
    decision: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_sections: List[str] = field(default_factory=list)


class ConversationSession:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.turns: List[ConversationTurn] = []
        self.current_order_id: Optional[str] = None
        self.current_customer_id: Optional[str] = None
        self.current_intent: Optional[str] = None
        self.active_items: List[Dict[str, Any]] = []
        self.verified: bool = False
        self.retry_attempts: int = 0
        self.metadata: Dict[str, Any] = {}

    def add_turn(self, user_msg: str, assistant_reply: str, decision: str = "answer",
                 tool_calls: Optional[List[Dict[str, Any]]] = None,
                 evidence: Optional[List[str]] = None) -> ConversationTurn:
        turn = ConversationTurn(
            turn_index=len(self.turns) + 1,
            user_message=user_msg,
            assistant_reply=assistant_reply,
            decision=decision,
            tool_calls=tool_calls or [],
            retrieved_sections=evidence or []
        )
        self.turns.append(turn)
        return turn

    def get_history_as_messages(self) -> List[Dict[str, str]]:
        msgs = []
        for turn in self.turns:
            msgs.append({"role": "user", "content": turn.user_message})
            msgs.append({"role": "assistant", "content": turn.assistant_reply})
        return msgs
