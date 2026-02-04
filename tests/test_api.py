import httpx

import asyncio
from datetime import datetime

BASE_URL = "http://localhost:8001"

async def test_health():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        print("✓ Health Check Passed")

async def test_init():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/init?village_id=wayanad_meppadi")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "map_config" in data
        assert "boundary" in data
        assert "infrastructure" in data
        print("✓ Init Endpoint Passed")

async def test_simulate():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/simulate?rainfall=100&village_id=wayanad_meppadi")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "simulation" in data
        assert data["simulation"]["type"] == "FeatureCollection"
        print("✓ Simulation Endpoint Passed")

async def test_rescue():
    async with httpx.AsyncClient() as client:
        # Using coordinates near the center of Wayanad Meppadi
        response = await client.get(f"{BASE_URL}/api/rescue?lat=11.555&lon=76.135&village_id=wayanad_meppadi")
        assert response.status_code == 200
        data = response.json()
        # It might fail if no path found, but the endpoint should return 200
        assert "rescue_path" in data
        print("✓ Rescue Endpoint Passed")

async def main():
    print("Starting API Tests...")
    try:
        await test_health()
        await test_init()
        await test_simulate()
        await test_rescue()
        print("\nAll Tests Passed Successfully!")
    except Exception as e:
        print(f"\n❌ Test Failed: {str(e)}")
        # Print more details if it's an HTTP error
        if isinstance(e, AssertionError):
            print("Assertion failed.")

if __name__ == "__main__":
    asyncio.run(main())
