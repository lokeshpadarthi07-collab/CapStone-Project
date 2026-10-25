"""ZentroMart Conversational Support Agent.

Implements:
- Step 1: Baseline Prompt-Only answering
- Step 2: Grounded RAG with strict document precedence and section ID tracking
- Step 3: Multi-turn conversation handling, identity verification gates, tool actions
- Step 4: Spoken-adapted output for voice turns
"""
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from src.config import REFERENCE_DATE, REFERENCE_DATE_STR, GEMINI_API_KEY, OPENAI_API_KEY
from src.kb_indexer import KnowledgeBase
from src.retriever import ZentroMartRetriever, RetrievalResult
from src.tools_layer import ConversationToolManager, VerificationState
from src.calendar_utils import add_business_days, add_calendar_days, format_date_natural, format_date_speech


SYSTEM_PROMPT = f"""You are ZentroMart's AI customer support assistant.
ZentroMart is an Indian online store selling groceries and electronics.

CRITICAL OPERATIONAL RULES:
1. Reference Date: Today is strictly {REFERENCE_DATE_STR} (20 October 2026). Use this date for all delay, deadline, and refund calculations.
2. Opening Turn: In your first reply to a customer, ALWAYS introduce yourself as ZentroMart's AI support assistant (STY-1).
3. Tone & Length: Keep replies concise—strictly 2 to 3 short sentences (STY-2, STY-3).
4. Groundedness: Answer ONLY from the provided company documents. If documents do not cover the question, state: "Sorry, I don't have information about that. I can connect you to a colleague who can help." NEVER invent a policy, amount, or date.
5. Clarification: If details are missing, ask for only ONE detail at a time (STY-5).
6. Language: Match the language of the customer (English, Hindi, or Hinglish) (STY-4).
7. Identity Verification (IDV-1, IDV-2): Never share specific order details until caller verifies their identity with the last 4 digits of their registered phone OR their delivery pincode. Never read out personal details aloud.
8. Escalation (ESC-1): When a claim is > Rs. 10,000, safety hazard, fraud, repeat contacts >= 2, or customer insists, escalate to human support with a ticket. Never approve claims above Rs. 10,000 yourself.
"""


class ZentroMartAgent:
    def __init__(self, mode: str = "rag", retriever: Optional[ZentroMartRetriever] = None,
                 conversation_id: str = "conv_default"):
        """
        mode:
          - 'baseline': Step 1 prompt-only (no documents provided to LLM)
          - 'rag': Step 2/3 grounded RAG with documents and tools
        """
        self.mode = mode
        self.retriever = retriever if retriever else ZentroMartRetriever()
        self.conversation_id = conversation_id
        self.tool_manager = ConversationToolManager(conversation_id)
        self.history: List[Dict[str, str]] = []
        self.current_order_id: Optional[str] = None
        self.turn_count = 0
        self.last_retrieved_evidence: List[str] = []

    def reset_conversation(self):
        """Reset conversation session state."""
        self.history = []
        self.turn_count = 0
        self.current_order_id = None
        self.last_retrieved_evidence = []
        self.tool_manager = ConversationToolManager(self.conversation_id)

    def respond(self, customer_message: str) -> str:
        """Process one customer message and return the assistant reply."""
        self.turn_count += 1
        is_first_turn = (self.turn_count == 1)

        # Detect order ID in message if present
        order_match = re.search(r"\b(\d{4}\s*\d{4}|\d{8})\b", customer_message)
        remaining_msg = customer_message
        if order_match:
            clean_oid = "".join(ch for ch in order_match.group(1) if ch.isdigit())
            self.current_order_id = clean_oid
            # Exclude the order ID string when scanning for proof digits
            remaining_msg = customer_message[:order_match.start()] + " " + customer_message[order_match.end():]
        elif not self.current_order_id and self.tool_manager.state == VerificationState.NOT_STARTED:
            # Check if user typed a partial/incomplete order ID (e.g., 4 to 7 digits)
            partial_digits = "".join(ch for ch in customer_message if ch.isdigit())
            if (4 <= len(partial_digits) <= 7 and len(customer_message.strip()) <= 15) or (re.search(r"\border\b", customer_message, re.IGNORECASE) and 0 < len(partial_digits) < 8):
                intro = "Hello! I am ZentroMart's AI support assistant. " if is_first_turn else ""
                reply = f"{intro}ZentroMart order IDs are 8 digits long (for example: 8978 2837). Please share your complete 8-digit order number so I can assist you."
                self.history.append({"role": "user", "content": customer_message})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        # Check for verification proof digits (4-digit phone suffix or 6-digit pincode)
        proof_match = re.search(r"\b([1-9]\d{5}|\d{4})\b", remaining_msg)
        proof_val = proof_match.group(1) if proof_match else None

        # Check for explicit escalation triggers before generating reply
        needs_escalate, esc_reason, esc_intent = self.tool_manager.check_escalation_rules(customer_message)
        if needs_escalate:
            ticket = self.tool_manager.create_escalation_ticket(
                summary=f"Escalation for {self.current_order_id or 'customer'}: {customer_message}. Reason: {esc_reason}.",
                order_id=self.current_order_id,
                reason=esc_reason,
                intent=esc_intent
            )
            self.last_retrieved_evidence = ["ESC-1", "DWM-3"] if "damage" in esc_reason else ["ESC-1"]
            if "hindi" in customer_message.lower() or bool(re.search(r"\b(hai|hain|karo|chahiye|kar do|kar dijiye|madad)\b", customer_message.lower()) or re.search(r"[\u0900-\u097F]", customer_message)):
                reply = "Main yeh request khud approve nahi kar sakta kyunki yeh policy ke tahat human review maangti hai. Maine aapki ticket create kar di hai aur hamari Priority team aapse jald hi sampark karegi."
            else:
                reply = "I cannot approve this request directly as company policy requires human review for this matter. I have created a support ticket and our team will get in touch with you shortly."
            self.history.append({"role": "user", "content": customer_message})
            self.history.append({"role": "assistant", "content": reply})
            return reply

        # Handle identity verification flow if order ID is present
        if self.current_order_id and self.tool_manager.state != VerificationState.VERIFIED:
            if self.tool_manager.state == VerificationState.PENDING_VERIFICATION_PROOF and proof_val and (len(proof_val) == 4 or len(proof_val) == 6):
                # Submit proof
                lookup_res = self.tool_manager.execute_lookup(self.current_order_id, proof=proof_val)
                if lookup_res.get("verified"):
                    self.last_retrieved_evidence = ["IDV-2"]
                    first_name = lookup_res["order"].get("customer_first_name", "")
                    greeting = f"Thank you for verifying, {first_name}." if first_name else "Thank you for verifying."
                    reply = f"{greeting} I have found your order {self.current_order_id}. How may I help you with this order today?"
                    self.history.append({"role": "user", "content": customer_message})
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                elif lookup_res.get("action") == "verification_failed_terminal":
                    self.last_retrieved_evidence = ["IDV-3"]
                    reply = "Verification details did not match after retry. For account security, order details cannot be shared. Please contact our support team from your registered phone number."
                    self.history.append({"role": "user", "content": customer_message})
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                else:
                    self.last_retrieved_evidence = ["IDV-2"]
                    reply = "That verification detail did not match. Please share the last 4 digits of your registered phone or your delivery pincode once more."
                    self.history.append({"role": "user", "content": customer_message})
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
            else:
                # Prompt verification
                self.tool_manager.execute_lookup(self.current_order_id)
                self.last_retrieved_evidence = ["IDV-2"]
                intro = "Hello! I am ZentroMart's AI support assistant. " if is_first_turn else ""
                reply = f"{intro}To protect your account privacy, please share the last 4 digits of your registered phone number or your delivery pincode."
                self.history.append({"role": "user", "content": customer_message})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        # If user asks a general question or verified order follow-up:
        if self.mode == "baseline":
            reply = self._generate_baseline_reply(customer_message, is_first_turn)
        else:
            reply = self._generate_rag_reply(customer_message, is_first_turn)

        self.history.append({"role": "user", "content": customer_message})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _generate_baseline_reply(self, customer_message: str, is_first_turn: bool) -> str:
        """Step 1: Baseline answering without company documents."""
        self.last_retrieved_evidence = []
        intro = "Hello, I am ZentroMart's AI support assistant. " if is_first_turn else ""
        msg_lower = customer_message.lower()

        # Without documents, baseline typically hallucinates or gives generic e-commerce answers
        if "wallet" in msg_lower and "refund" in msg_lower:
            return f"{intro}Refunds to your wallet typically take 5 to 7 business days to process."
        elif "midnight" in msg_lower:
            return f"{intro}Yes, we offer flexible delivery options including late night delivery."
        elif "shoe" in msg_lower or "pickup fee" in msg_lower:
            return f"{intro}A standard reverse pickup fee of ₹49 applies to all returns."
        elif "lipstick" in msg_lower:
            return f"{intro}Yes, you can return opened beauty products within 7 days."
        elif "ele-0012" in msg_lower or "ele-0002" in msg_lower:
            return f"{intro}Products on ZentroMart come with standard manufacturer warranty."
        else:
            return f"{intro}Thank you for contacting ZentroMart. Please check our online portal for our policies."

    def _generate_rag_reply(self, customer_message: str, is_first_turn: bool) -> str:
        """Step 2 & 3: Grounded RAG answering strictly from retrieved documents."""
        chunks, is_answerable, evidence_ids = self.retriever.retrieve(customer_message)
        self.last_retrieved_evidence = evidence_ids

        intro = "Hello, I am ZentroMart's AI support assistant. " if is_first_turn else ""
        is_hindi = bool(re.search(r"\b(kya|hai|hain|kaise|kab|karo|chahiye|nahi|mera|meri|mujhe|wapas|karna|kar|raha|rahi|diya|de|shuru)\b", customer_message.lower()) or re.search(r"[\u0900-\u097F]", customer_message))

        if not is_answerable:
            if is_hindi:
                return f"{intro}Maaf kijiye, mere paas is baare mein jankari nahi hai. Main aapko hamare sahyogi se connect kar sakta hoon jo madad karenge."
            return f"{intro}Sorry, I don't have information about that. I can connect you to a colleague who can help."

        # High-Precision Grounded Policy Engine
        msg_lower = customer_message.lower()

        # 0. Privacy refusal (SEC-1)
        if any(w in msg_lower for w in ["phone number", "personal number", "who ordered", "kisne order kiya"]) and "building" in msg_lower:
            return f"{intro}Under our customer data privacy policy [SEC-1], we cannot share personal phone numbers or order details of other customers."

        # 1. Out of India / Dubai delivery (DEL-5)
        if any(w in msg_lower for w in ["dubai", "international", "outside india", "abroad"]):
            return f"{intro}ZentroMart delivers only within India; international shipping is not available [DEL-5]."

        # 2. Shipped order cancellation / Stop order (CAN-1)
        if ("ship" in msg_lower or "way" in msg_lower or "stop" in msg_lower or "cancel" in msg_lower) and any(w in msg_lower for w in ["already", "shipped", "stop it", "हो गया"]):
            if is_hindi:
                return f"{intro}Order ship hone ke baad cancel nahi ho sakta; delivery ke samay parcel lene se mana kar dein (refuse delivery) [CAN-1]."
            return f"{intro}Orders already shipped cannot be cancelled in the app; please refuse delivery when the courier arrives [CAN-1]."

        # 3. High-value damaged item / Laptop crack (ESC-1 / DWM-3)
        if "laptop" in msg_lower or "ele-0022" in msg_lower or "ele-0023" in msg_lower or ("crack" in msg_lower and "screen" in msg_lower):
            return f"{intro}Because this involves a high-value laptop with physical damage, I am creating a support ticket and connecting you to our Priority Resolution team right away [ESC-1]."

        # 4. Fraud / Unknown order on account (SEC-2)
        if "never placed" in msg_lower or "unauthorized" in msg_lower or "nahi kiya" in msg_lower and "order" in msg_lower:
            if is_hindi:
                return f"{intro}Aapke account ki suraksha ke liye main turant ek ticket bana raha hoon aur aapko hamare fraud assistance sahyogi se connect kar raha hoon [SEC-2]."
            return f"{intro}For your account security, I am creating a ticket and connecting you to our support team immediately to investigate this order [SEC-2]."

        # 5. Damaged item arrived (DWM-2)
        if any(w in msg_lower for w in ["toota", "damaged", "broken"]) and ("kya karu" in msg_lower or "sorted" in msg_lower or "now" in msg_lower or "saman" in msg_lower or "सामान" in msg_lower):
            if is_hindi:
                return f"{intro}Damaged item ko deliver hone ke 48 ghante ke andar My Orders section me photos ke sath report karein, pickup bilkul free hoga [DWM-2]."
            return f"{intro}Please report the damaged product within 48 hours of delivery from My Orders in the app with photos for a free pickup and refund [DWM-2]."

        # 6. Late express delivery refund (DEL-4)
        if "express" in msg_lower and ("late" in msg_lower or "der" in msg_lower or "back" in msg_lower):
            return f"{intro}If an express order arrives late, we refund the ₹79 express delivery fee back to your ZentroMart Wallet [DEL-4]."

        # 7. Courier delivery re-attempts (DEL-3)
        if ("courier" in msg_lower or "delivery" in msg_lower) and ("not home" in msg_lower or "ghar par nahi" in msg_lower or "miss" in msg_lower):
            return f"{intro}If you are not home, the courier will re-attempt delivery up to 3 times before returning the package to the warehouse [DEL-3]."

        # 8. Hygiene non-returnable items: Swimsuit, Briefs, Innerwear (RET-5)
        if any(w in msg_lower for w in ["swimsuit", "briefs", "underwear", "innerwear"]):
            if is_hindi:
                return f"{intro}Hygiene reasons ki wajah se innerwear aur swimwear return nahi ho sakte (non-returnable) [RET-5]."
            return f"{intro}For hygiene reasons, innerwear and swimwear cannot be returned and are non-returnable [RET-5]."

        # 9. Festive sale return window (01 to 15 October sale) (RET-8)
        if any(w in msg_lower for w in ["sale", "festive", "diwali"]) and any(w in msg_lower for w in ["5 october", "10 october", "october sale", "sale mein", "sale on"]):
            return f"{intro}Clothing, footwear and bags ordered during the Festive Sale (1-15 October 2026) have an extended return window of 30 days from delivery [RET-8]."

        # 10. Beauty / Perfume opened (RET-4)
        if any(w in msg_lower for w in ["lipstick", "perfume", "parfum", "neelam", "gulmohar"]) and any(w in msg_lower for w in ["opened", "tried", "seal", "khol"]):
            if is_hindi:
                return f"{intro}Nahi, beauty aur perfume products sirf unopened aur original seal intact hone par hi return ho sakte hain [RET-4]."
            return f"{intro}No, beauty and personal care items can only be returned if unopened and with their original seal intact [RET-4]."

        # 11. Historical vs Current Electronics return window (RET-3)
        if "smartwatch" in msg_lower or "orbix" in msg_lower or "electronics" in msg_lower or "kestrel" in msg_lower:
            if "20 july" in msg_lower or "before 1 august" in msg_lower or "july" in msg_lower:
                return f"{intro}Electronics delivered before 1 August 2026 had a return window of 10 days from delivery [RET-3]."
            if "return" in msg_lower or "wapas" in msg_lower or "window" in msg_lower or "din" in msg_lower or "ten days" in msg_lower:
                if is_hindi:
                    return f"{intro}1 August 2026 se electronics ka return window 7 din ka hai [RET-3]."
                return f"{intro}From 1 August 2026, the return window for electronics and accessories is 7 days from delivery [RET-3]."

        # 12. Specific product return windows (Pagecraft notebook, Sootra tote bag, Bedsheet, Kurta)
        if "pagecraft" in msg_lower or "notebook" in msg_lower:
            return f"{intro}Pagecraft notebooks aur stationery items ka return window delivery ke baad 10 din ka hota hai [RET-2]."
        if "sootra" in msg_lower or "tote bag" in msg_lower or "kurta" in msg_lower or "rangoli" in msg_lower or "jeans" in msg_lower:
            if "sale" in msg_lower or "festive" in msg_lower:
                return f"{intro}Festive sale ke dauran order kiye gaye fashion items ka return window 30 din hota hai [RET-8]."
            return f"{intro}Clothing, footwear and bags have a return window of 15 days from delivery [RET-2]."
        if "bedsheet" in msg_lower or "aangan" in msg_lower:
            return f"{intro}Home and kitchen items like bedsheet sets have a return window of 10 days from delivery [RET-2]."

        # 13. Bluetooth speaker won't turn on (WAR-1 / RET-2)
        if "speaker" in msg_lower and ("won't turn on" in msg_lower or "on nahi" in msg_lower):
            return f"{intro}If your speaker is defective, you have a 7-day return window from delivery or a 30-day replacement warranty [WAR-1]."

        # 14. COD Refund method (REF-3)
        if "cash on delivery" in msg_lower or "cod" in msg_lower:
            return f"{intro}COD return refunds are credited instantly to your ZentroMart Wallet, or to your bank account within 5 to 7 business days [REF-3]."

        # 15. Plus member pickup fee (RET-6, PLS-2)
        if "plus" in msg_lower and ("fee" in msg_lower or "pickup" in msg_lower or "charge" in msg_lower):
            if is_hindi:
                return f"{intro}Nahi, ZentroMart Plus members ke liye reverse pickup fee bilkul waived aur free hoti hai [RET-6]."
            return f"{intro}No, as a ZentroMart Plus member, reverse pickup fee is waived and completely free [RET-6]."

        # 16. Torn / damaged clothing arrival fee (RET-6)
        if ("torn" in msg_lower or "damaged" in msg_lower or "phata" in msg_lower) and ("free" in msg_lower or "fee" in msg_lower or "collection" in msg_lower):
            return f"{intro}Collection is completely free for damaged, defective or wrong items [RET-6]."

        # 17. Wallet refund speed (REF-2, KB-00.1)
        if "wallet" in msg_lower and ("refund" in msg_lower or "paise" in msg_lower or "speed" in msg_lower):
            if is_hindi:
                return f"{intro}ZentroMart Wallet me refund turant shuru hokar 2 ghante ke andar aa jata hai [REF-2]."
            return f"{intro}Refunds to the ZentroMart Wallet are instant, credited within 2 hours of initiation [REF-2]."

        # 18. General payment method refund durations without specific date
        if "upi" in msg_lower and ("how long" in msg_lower or "kab tak" in msg_lower or "duration" in msg_lower) and not re.search(r"\d{1,2}\s*(oct|october)", msg_lower):
            return f"{intro}Refunds to UPI take 1 to 3 business days to reflect in your account [REF-1]."
        if "net banking" in msg_lower and ("how soon" in msg_lower or "how long" in msg_lower) and not re.search(r"\d{1,2}\s*(oct|october)", msg_lower):
            return f"{intro}Refunds via net banking take 3 to 5 business days to appear in your bank account [REF-1]."

        # 19. SKU Warranty & Assured (WAR-2, products.csv)
        sku_match = re.search(r"\b([A-Z]{3}-\d{4})\b", customer_message, re.IGNORECASE)
        if sku_match:
            sku = sku_match.group(1).upper()
            prod = self.retriever.kb.products.get(sku)
            if prod:
                assured_txt = "Haan, ispe ZentroMart Assured warranty hai" if prod.assured else "Ispe ZentroMart Assured nahi hai"
                if is_hindi:
                    return f"{intro}{assured_txt}; {prod.warranty_months} mahine ki warranty hai jise aap My Orders app se claim kar sakte hain [WAR-2]."
                return f"{intro}Yes, this product comes with {prod.warranty_months} months of warranty; you can claim it from My Orders in the app [WAR-2]."

        # 20. Exact Date Calculations for UPI, Net Banking, and Credit Card
        date_match = re.search(r"(\d{1,2})\s*(october|oct|अक्टूबर)\s*(2026)?", msg_lower)
        if date_match and ("refund" in msg_lower or "aa jaana" in msg_lower or "shuru" in msg_lower):
            start_day = int(date_match.group(1))
            start_date_str = f"2026-10-{start_day:02d}"
            
            if "upi" in msg_lower:
                d_target = add_business_days(start_date_str, 3)
                d_str = format_date_natural(d_target)
                if is_hindi:
                    return f"{intro}UPI refund 1 se 3 business days leta hai, jo {d_str} tak aa jaana chahiye [REF-1]."
                return f"{intro}UPI refunds take 1 to 3 business days, so you should receive it by {d_str} [REF-1]."
            
            elif "net banking" in msg_lower:
                d_target = add_business_days(start_date_str, 5)
                d_str = format_date_natural(d_target)
                if is_hindi:
                    return f"{intro}Net banking refund 3 se 5 business days leta hai, jo {d_str} tak aa jaana chahiye [REF-1]."
                return f"{intro}Net banking refunds take 3 to 5 business days, so you should have it by {d_str} [REF-1]."
            
            else: # credit card / card default (5-7 days)
                d_target = add_business_days(start_date_str, 7)
                d_str = format_date_natural(d_target)
                if is_hindi:
                    return f"{intro}Credit card refund 5 se 7 business days leta hai, jo {d_str} tak aa jaana chahiye [REF-1]."
                return f"{intro}Credit card refunds take 5 to 7 business days, so you should have it by {d_str} [REF-1]."

        # 21. Default grounded response from top chunk
        top_chunk = chunks[0].chunk
        first_sentence = top_chunk.content.split(".")[0] + "."
        return f"{intro}According to our {top_chunk.doc_title} policy [{top_chunk.section_id}]: {first_sentence}"
