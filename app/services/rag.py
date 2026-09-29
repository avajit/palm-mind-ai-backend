import json
import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.services.llm import LLMClient
from app.services.memory import MemoryService
from app.services.embeddings import EmbeddingService
from app.services.vector_store import QdrantService
from app.db import BookingRecord


class RAGService:
    """
    Orchestrates manual Custom RAG, multi-turn history rewriting,
    intent routing, and interview booking state management.
    """

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.memory = MemoryService()
        self.embedder = EmbeddingService()
        self.qdrant = QdrantService()

    def process_message(
        self, session_id: str, message: str, db: Session
    ) -> Dict[str, Any]:
        """
        Main entry point for handling user chat messages.
        Steps:
        1. Load chat history & booking state from Redis.
        2. Run LLM router / analyzer to determine intent, rewrite query, extract booking fields.
        3. If intent == 'question': execute RAG pipeline with vector search and history.
        4. If intent == 'booking': validate/collect booking slots, save to SQL when complete.
        5. Update chat history in Redis.
        6. Return structured response.
        """
        # 1. Load history and booking state
        history = self.memory.get_history(session_id)
        booking_state = self.memory.get_booking_state(session_id)
        today_str = datetime.now().strftime("%Y-%m-%d")

        # 2. Router / Analyzer Prompt
        router_system_prompt = f"""You are an intelligent assistant router and analyzer for an AI document assistant & booking system.
Today's date is {today_str}.

Analyze the user's latest message considering the chat history and current in-progress booking state.
Return a JSON object with EXACTLY these keys:
- "intent": either "question" (asking a question about documents or general info) or "booking" (wanting to schedule/book an interview).
- "standalone_query": a rephrased version of the latest user message that incorporates context from chat history so it is fully self-contained.
- "name": extracted candidate name if mentioned, otherwise null or existing booking value.
- "email": extracted email address if mentioned, otherwise null or existing booking value.
- "date": extracted date in YYYY-MM-DD format if mentioned (resolve relative dates like "tomorrow" or "next Monday" using today's date {today_str}), otherwise null or existing.
- "time": extracted time in HH:MM 24-hour format if mentioned, otherwise null or existing.

Current in-progress booking state: {json.dumps(booking_state)}

Respond ONLY with valid JSON. No markdown code fences, no extra text."""

        messages_for_router = [{"role": "system", "content": router_system_prompt}]
        # Add last 4 messages of history for context
        for h in history[-4:]:
            messages_for_router.append(h)
        messages_for_router.append({"role": "user", "content": message})

        router_raw = self.llm.chat(messages_for_router, temperature=0.1)
        parsed_routing = self._parse_json_safe(router_raw)

        intent = parsed_routing.get("intent", "question")
        standalone_query = parsed_routing.get("standalone_query", message)

        sources: List[Dict[str, Any]] = []
        answer = ""
        booking_result: Optional[Dict[str, Any]] = None

        if intent == "booking":
            # Extract and update booking state
            new_fields = {
                "name": parsed_routing.get("name") or booking_state.get("name"),
                "email": parsed_routing.get("email") or booking_state.get("email"),
                "date": parsed_routing.get("date") or booking_state.get("date"),
                "time": parsed_routing.get("time") or booking_state.get("time"),
            }
            # Filter out None or empty string
            new_fields = {k: v for k, v in new_fields.items() if v and v != "null"}
            current_booking = self.memory.update_booking_state(session_id, new_fields)

            # Validate fields
            missing = []
            invalid_reasons = []

            if not current_booking.get("name"):
                missing.append("full name")

            email = current_booking.get("email")
            if not email:
                missing.append("email address")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                invalid_reasons.append("Invalid email format.")

            b_date = current_booking.get("date")
            if not b_date:
                missing.append("date (YYYY-MM-DD)")
            else:
                try:
                    parsed_date = datetime.strptime(b_date, "%Y-%m-%d").date()
                    if parsed_date < date.today():
                        invalid_reasons.append("Date cannot be in the past.")
                except ValueError:
                    invalid_reasons.append("Date must be in YYYY-MM-DD format.")

            b_time = current_booking.get("time")
            if not b_time:
                missing.append("time (HH:MM)")
            else:
                try:
                    datetime.strptime(b_time, "%H:%M")
                except ValueError:
                    invalid_reasons.append("Time must be in HH:MM 24-hour format.")

            if invalid_reasons:
                answer = f"I noticed some issues with your booking details: {', '.join(invalid_reasons)}. Please provide corrected information."
                booking_result = {
                    "status": "invalid",
                    "name": current_booking.get("name"),
                    "email": current_booking.get("email"),
                    "date": current_booking.get("date"),
                    "time": current_booking.get("time"),
                    "errors": invalid_reasons,
                }
            elif missing:
                answer = f"I'd love to schedule your interview! To proceed, please provide your {', '.join(missing)}."
                booking_result = {
                    "status": "collecting",
                    "name": current_booking.get("name"),
                    "email": current_booking.get("email"),
                    "date": current_booking.get("date"),
                    "time": current_booking.get("time"),
                    "missing": missing,
                }
            else:
                # All fields present and valid -> Save to SQLite!
                booking_record = BookingRecord(
                    name=current_booking["name"],
                    email=current_booking["email"],
                    date=current_booking["date"],
                    time=current_booking["time"],
                )
                db.add(booking_record)
                db.commit()
                db.refresh(booking_record)

                answer = f"Success! Your interview is booked for {current_booking['date']} at {current_booking['time']} under {current_booking['name']} ({current_booking['email']}). We look forward to speaking with you!"
                booking_result = {
                    "status": "confirmed",
                    "booking_id": booking_record.id,
                    "name": current_booking.get("name"),
                    "email": current_booking.get("email"),
                    "date": current_booking.get("date"),
                    "time": current_booking.get("time"),
                }
                self.memory.clear_booking_state(session_id)

        else:
            # intent == "question" (Custom RAG)
            # 1. Embed standalone query
            query_vector = self.embedder.generate_embeddings([standalone_query])[0]

            # 2. Retrieve top-k chunks from Qdrant
            retrieved_chunks = self.qdrant.search(query_vector, limit=4)

            # 3. Build context string
            context_blocks = []
            for i, chunk in enumerate(retrieved_chunks):
                sources.append({
                    "document_id": chunk.get("document_id"),
                    "filename": chunk.get("filename"),
                    "chunk_index": chunk.get("chunk_index"),
                    "score": chunk.get("score"),
                })
                context_blocks.append(f"[Source {i+1} - {chunk.get('filename')}]:\n{chunk.get('text')}")

            context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."

            # 4. Construct RAG prompt
            rag_system_prompt = """You are a helpful and precise AI assistant answering questions based on provided document context.
Follow these rules:
1. Answer the question using ONLY the provided document context.
2. If the answer cannot be found in the context, explicitly state: "I cannot find the answer in the provided documents."
3. Be professional, concise, and accurate."""

            rag_messages = [{"role": "system", "content": rag_system_prompt}]
            # Add recent chat history for multi-turn conversation awareness
            for h in history[-6:]:
                rag_messages.append(h)

            user_prompt = f"""Context from documents:
{context_str}

User Question: {standalone_query}"""
            rag_messages.append({"role": "user", "content": user_prompt})

            answer = self.llm.chat(rag_messages, temperature=0.3)

        # 5. Update Redis chat history
        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", answer)

        return {
            "session_id": session_id,
            "intent": intent,
            "standalone_query": standalone_query,
            "answer": answer,
            "sources": sources,
            "booking": booking_result,
        }

    def _parse_json_safe(self, raw_text: str) -> Dict[str, Any]:
        """Safely parses JSON from LLM output, stripping code fences if present."""
        cleaned = raw_text.strip()
        # Remove markdown code blocks if any
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"intent": "question", "standalone_query": raw_text}
