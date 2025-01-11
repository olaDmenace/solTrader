import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock
from src.config.settings import Settings

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Add src directory to Python path
src_path = os.path.join(project_root, "src")
sys.path.append(str(src_path))

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    
    # Clean up pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

@pytest.fixture(autouse=True)
async def cleanup_aiohttp():
    """Cleanup any pending aiohttp sessions after each test"""
    yield
    await asyncio.sleep(0.1)  # Allow pending tasks to complete
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass