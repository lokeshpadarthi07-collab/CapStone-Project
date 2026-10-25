"""Order service handling identity-checked lookups and escalation tickets.

Integrates directly with the actual tools/tools.py functions without modifying signatures.
"""
from typing import Dict, Any, Optional
from src.tools_layer import ConversationToolManager, VerificationState


class OrderService:
    def __init__(self, conversation_id: str = "default"):
        self.manager = ConversationToolManager(conversation_id)

    def lookup_order(self, order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
        """Lookup order enforcing two-phase caller verification."""
        return self.manager.execute_lookup(order_id, proof=verification_value)

    def create_ticket(self, summary: str, order_id: Optional[str] = None,
                      reason: Optional[str] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        """Create support ticket. Returns created=True on success."""
        return self.manager.create_escalation_ticket(summary=summary, order_id=order_id, reason=reason, intent=intent)

    @property
    def is_verified(self) -> bool:
        return self.manager.state == VerificationState.VERIFIED

    @property
    def verified_order(self) -> Optional[Dict[str, Any]]:
        return self.manager.verified_order
