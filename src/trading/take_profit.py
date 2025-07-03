from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ScaledTakeProfit:
    def __init__(self, entry_price: float, position_size: float):
        self.entry_price = entry_price
        self.total_size = position_size
        self.levels: List[Dict[str, Any]] = []
        self.executed_levels: List[int] = []
        self.completed = False
        
    def add_level(self, percentage: float, size_fraction: float) -> None:
        """Add a new take profit level with validation"""
        # Validate size fractions don't exceed 100%
        total_size = sum(level['size'] / self.total_size for level in self.levels) + size_fraction
        if total_size > 1.0:
            raise ValueError("Total size fractions cannot exceed 100%")
            
        self.levels.append({
            'price': self.entry_price * (1 + percentage),
            'size': self.total_size * size_fraction,
            'percentage': percentage,
            'executed': False,
            'execute_time': None
        })
        self.levels.sort(key=lambda x: x['percentage'])

    def check_levels(self, current_price: float) -> Optional[Dict[str, Any]]:
        """Check if any take profit levels are triggered"""
        if self.completed:
            return None
            
        for i, level in enumerate(self.levels):
            if i not in self.executed_levels and current_price >= level['price']:
                self.executed_levels.append(i)
                level['executed'] = True
                level['execute_time'] = datetime.now()
                
                # Check if all levels completed
                if len(self.executed_levels) == len(self.levels):
                    self.completed = True
                    
                return {
                    'price': level['price'],
                    'size': level['size'],
                    'percentage': level['percentage']
                }
        return None

    @classmethod
    def create_default_levels(cls, entry_price: float, position_size: float) -> 'ScaledTakeProfit':
        """Create instance with default profit targets"""
        tp = cls(entry_price=entry_price, position_size=position_size)
        # Default strategy: 50% at 2%, 30% at 5%, 20% at 8%
        tp.add_level(0.02, 0.5)  # First target
        tp.add_level(0.05, 0.3)  # Second target 
        tp.add_level(0.08, 0.2)  # Final target
        return tp
        
    def get_active_levels(self) -> List[Dict[str, Any]]:
        """Get all untriggered take profit levels"""
        return [
            {
                'price': level['price'],
                'size': level['size'],
                'percentage': level['percentage']
            }
            for i, level in enumerate(self.levels)
            if i not in self.executed_levels
        ]
        
    def get_next_target(self) -> Optional[float]:
        """Get the next untriggered take profit target"""
        for i, level in enumerate(self.levels):
            if i not in self.executed_levels:
                return level['price']
        return None
        
    def get_completion_percentage(self) -> float:
        """Calculate percentage of position that has been closed"""
        closed_size = sum(
            level['size'] for i, level in enumerate(self.levels)
            if i in self.executed_levels
        )
        return (closed_size / self.total_size) * 100.0

    def reset(self) -> None:
        """Reset all levels to untriggered state"""
        self.executed_levels.clear()
        self.completed = False
        for level in self.levels:
            level['executed'] = False
            level['execute_time'] = None