import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/recognize"
    
    async with websockets.connect(uri) as websocket:
        while True:
            response = await websocket.recv()
            print("Received:", json.loads(response))

asyncio.run(test_websocket())
