from fastapi.testclient import TestClient
from app.main import app
import uuid

client = TestClient(app)

def run_tests():
    print("=== STARTING COMPREHENSIVE QA TEST SUITE ===")

    # 1. Test Health Check
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("[PASS] 1. Health Check (/health)")

    # 2. Test Ingestion with a sample text file
    sample_text = "Palm Mind AI builds intelligent conversational assistants. Our headquarters are in San Francisco. The CEO is Jane Doe."
    files = {"file": ("test_doc.txt", sample_text.encode("utf-8"), "text/plain")}
    data = {"chunking_strategy": "sentence"}
    resp = client.post("/ingest", files=files, data=data)
    assert resp.status_code == 200, f"Ingestion failed: {resp.text}"
    ingest_json = resp.json()
    print(f"[PASS] 2. Document Ingestion (/ingest): document_id={ingest_json['document_id']}, chunks={ingest_json['chunk_count']}")

    # 3. Test RAG Question Flow
    session_id = f"qa-test-session-{uuid.uuid4()}"
    resp = client.post("/chat", json={"session_id": session_id, "message": "Where is Palm Mind AI's headquarters located?"})
    assert resp.status_code == 200, f"Chat question flow failed: {resp.text}"
    chat_json = resp.json()
    assert chat_json["intent"] == "question"
    assert "San Francisco" in chat_json["answer"]
    print(f"[PASS] 3. RAG Question Flow: Answer -> {chat_json['answer']}")

    # 4. Test Multi-Turn Query Rewriting
    resp = client.post("/chat", json={"session_id": session_id, "message": "Who is the CEO?"})
    assert resp.status_code == 200, f"Multi-turn chat failed: {resp.text}"
    multiturn_json = resp.json()
    assert "Jane Doe" in multiturn_json["answer"]
    print(f"[PASS] 4. Multi-Turn Query Rewriting: Answer -> {multiturn_json['answer']}")

    # 5. Test Booking Flow - Incomplete (Collecting state)
    booking_session = f"qa-booking-session-{uuid.uuid4()}"
    resp = client.post("/chat", json={"session_id": booking_session, "message": "I want to book an interview."})
    assert resp.status_code == 200
    b_json = resp.json()
    assert b_json["intent"] == "booking"
    assert b_json["booking"]["status"] == "collecting"
    print(f"[PASS] 5. Booking Flow (Collecting missing fields): {b_json['answer']}")

    # 6. Test Booking Flow - Invalid Email / Past Date
    resp = client.post("/chat", json={"session_id": booking_session, "message": "My name is Bob, email is invalid-email, date is 2020-01-01, time is 25:60"})
    assert resp.status_code == 200
    invalid_json = resp.json()
    assert invalid_json["booking"]["status"] == "invalid"
    print(f"[PASS] 6. Booking Flow Validation (Rejected invalid email/date/time): {invalid_json['answer']}")

    # 7. Test Booking Flow - Successful Completion & SQLite Storage
    resp = client.post("/chat", json={"session_id": booking_session, "message": "Let's fix that. Name is Bob Smith, email is bob@example.com, date is 2026-11-15, time is 14:00"})
    assert resp.status_code == 200
    success_json = resp.json()
    assert success_json["booking"]["status"] == "confirmed"
    assert success_json["booking"]["booking_id"] is not None
    print(f"[PASS] 7. Booking Flow Confirmation & SQLite Save (Booking ID: {success_json['booking']['booking_id']})")

    # 8. Edge Case: Empty message validation
    resp = client.post("/chat", json={"session_id": session_id, "message": "   "})
    assert resp.status_code == 400
    print("[PASS] 8. Edge Case Handling: Empty message correctly rejected with 400.")

    print("\n=== ALL QA TEST SCENARIOS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_tests()
