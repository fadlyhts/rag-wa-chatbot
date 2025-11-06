"""Test webhook endpoint"""

def test_webhook_incoming_message(client):
    """Test incoming message webhook"""
    payload = {
        "event": "message.incoming",
        "data": {
            "messageId": "msg_123",
            "from": "1234567890",
            "text": "Hello"
        }
    }
    
    response = client.post("/api/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "request_id" in data


def test_webhook_unsupported_event(client):
    """Test unsupported webhook event"""
    payload = {
        "event": "session.status",
        "data": {}
    }
    
    response = client.post("/api/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["acknowledged", "ignored"]
