# src/utils/persistence.py

import json
import aiofiles
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataPersistence:
    def __init__(self, base_path: str = "data/"):
        self.base_path = base_path
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists"""
        import os
        os.makedirs(self.base_path, exist_ok=True)

    async def save_data(self, data: Dict[str, Any], filename: str) -> bool:
        """Save data to JSON file"""
        try:
            filepath = f"{self.base_path}{filename}.json"
            async with aiofiles.open(filepath, 'w') as f:
                json_data = {
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                await f.write(json.dumps(json_data, indent=2))
            return True
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

    async def load_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            filepath = f"{self.base_path}{filename}.json"
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

class TradePersistence(DataPersistence):
    async def save_trade_history(self, trades: List[Dict[str, Any]]) -> bool:
        return await self.save_data({'trades': trades}, 'trade_history')

    async def load_trade_history(self) -> List[Dict[str, Any]]:
        data = await self.load_data('trade_history')
        return data.get('trades', []) if data else []

    async def save_paper_trades(self, trades: List[Dict[str, Any]]) -> bool:
        return await self.save_data({'paper_trades': trades}, 'paper_trades')

    async def load_paper_trades(self) -> List[Dict[str, Any]]:
        data = await self.load_data('paper_trades')
        return data.get('paper_trades', []) if data else []