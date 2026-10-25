"""Tool interface and caller identity verification state machine for ZentroMart.

Adheres strictly to:
- agent_identity_verification.md (IDV-1, IDV-2, IDV-3, IDV-4)
- agent_escalation_handoff.md (ESC-1, ESC-2, ESC-3, ESC-4)
- tools/tools.py API contracts
"""
import os
import sys
from typing import Dict, Any, Optional, Tuple

# Ensure root tools package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from tools.tools import lookup_order as _raw_lookup_order, create_ticket as _raw_create_ticket
except ImportError:
    from dataset.Capstone_Project_Dataset_FTTT_2026.tools.tools import (
        lookup_order as _raw_lookup_order,
        create_ticket as _raw_create_ticket,
    )


class VerificationState:
    NOT_STARTED = "not_started"
    PENDING_ORDER_ID = "pending_order_id"
    PENDING_VERIFICATION_PROOF = "pending_verification_proof"
    VERIFIED = "verified"
    FAILED = "failed"


class ConversationToolManager:
    """Manages identity verification and tool operations within a conversation session."""

    def __init__(self, conversation_id: str = "default"):
        self.conversation_id = conversation_id
        self.state = VerificationState.NOT_STARTED
        self.order_id: Optional[str] = None
        self.verified_order: Optional[Dict[str, Any]] = None
        self.retry_count = 0
        self.tickets_created: list = []

    def execute_lookup(self, order_id: str, proof: Optional[str] = None) -> Dict[str, Any]:
        """Execute lookup with verification state tracking."""
        clean_oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        self.order_id = clean_oid

        if proof is None or not str(proof).strip():
            # Phase 1: Initiating lookup
            res = _raw_lookup_order(clean_oid)
            self.state = VerificationState.PENDING_VERIFICATION_PROOF
            return {
                "action": "prompt_verification",
                "verification_required": True,
                "order_id": clean_oid,
                "message": res.get("message", "Please share the last 4 digits of your registered phone or your delivery pincode.")
            }

        # Phase 2: Verifying with phone last 4 or pincode
        clean_proof = "".join(ch for ch in str(proof) if ch.isdigit())
        res = _raw_lookup_order(clean_oid, verification_value=clean_proof)

        if res.get("verified") and res.get("found"):
            self.state = VerificationState.VERIFIED
            self.verified_order = res
            return {
                "action": "verified",
                "verified": True,
                "order": res
            }
        else:
            self.retry_count += 1
            if self.retry_count >= 2:
                self.state = VerificationState.FAILED
                return {
                    "action": "verification_failed_terminal",
                    "verified": False,
                    "retry_count": self.retry_count,
                    "message": "Verification failed twice. For security reasons, order details cannot be shared. Suggest contacting support from your registered phone."
                }
            else:
                self.state = VerificationState.PENDING_VERIFICATION_PROOF
                return {
                    "action": "verification_failed_retry",
                    "verified": False,
                    "retry_count": self.retry_count,
                    "message": "The verification details did not match. Please verify the last 4 digits of your phone or delivery pincode once more."
                }

    def check_escalation_rules(self, customer_message: str, intent: str = "") -> Tuple[bool, str, str]:
        """Check ESC-1 rules to determine if immediate human handover is required.

        Returns: (needs_escalation, reason, intent)
        """
        msg_lower = customer_message.lower()

        # 1. Product safety hazard
        if any(term in msg_lower for term in ["swelling", "battery swell", "sparks", "burning smell", "smoke", "electric shock", "injury"]):
            return True, "product_safety_hazard", "safety"

        # 2. Legal / consumer court / police threat
        if any(term in msg_lower for term in ["consumer court", "legal action", "lawyer", "police complaint", "fir", "sue you"]):
            return True, "legal_or_regulatory_complaint", "legal"

        # 3. Unauthorized order / account activity / fraud
        if any(term in msg_lower for term in ["didn't order", "did not order", "unauthorized", "fraud", "hacked", "stolen card"]):
            return True, "unauthorized_activity", "fraud"

        # 4. Explicit request for human agent
        if any(term in msg_lower for term in ["human agent", "talk to human", "real person", "senior agent", "supervisor", "manager"]):
            return True, "customer_requested_human", "human_request"

        # Check verified order specific rules
        if self.verified_order:
            # 5. Damaged, defective or wrong item worth > Rs. 10,000 (DWM-3, ESC-1.1)
            total_val = self.verified_order.get("total_inr", 0) or 0
            items = self.verified_order.get("items", [])
            max_item_val = max([it.get("unit_price_inr", 0) for it in items], default=total_val)

            if any(term in msg_lower for term in ["damaged", "broken", "toota", "defective", "wrong item", "damaged tv", "replacement"]):
                if max_item_val > 10000 or total_val > 10000:
                    return True, "high_value_damage", "damage"

            # 6. Repeat contact rule: contacts_last_7_days >= 2 (ESC-1.4)
            contacts = self.verified_order.get("contacts_last_7_days", 0) or 0
            if contacts >= 2:
                return True, "repeat_contact_threshold", "repeat_contact"

        return False, "", ""

    def create_escalation_ticket(self, summary: str, order_id: Optional[str] = None,
                                  reason: Optional[str] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        """Create ticket with compliance checks (ESC-2, ESC-3)."""
        oid = order_id or self.order_id or ""
        # Ensure summary meets length and structure criteria (>= 20 chars)
        if len(summary.strip()) < 25:
            summary = f"Customer issue requiring human review for order {oid}: {summary}. Reason: {reason or 'support escalation'}."

        res = _raw_create_ticket(summary=summary, order_id=oid, reason=reason, intent=intent)
        if res.get("created"):
            self.tickets_created.append(res)
        return res
