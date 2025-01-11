import logging
from typing import Dict, Any, List
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class MemeDetector:
    def __init__(self):
        self.meme_keywords: List[str] = [
            'doge', 'shib', 'pepe', 'inu', 'elon', 'moon', 'safe',
            'rocket', 'chad', 'wojak', 'based', 'cat', 'dog', 'baby',
            'king', 'queen', 'coin', 'floki', 'mars', 'moon', 'cute',
            'meme', 'web3', 'ai', 'meta'
        ]
        
        self.risk_keywords: List[str] = [
            'safe', 'fair', 'moon', 'elon', 'guaranteed', 'profit',
            'instant', 'rich', 'billion', 'million', 'airdrop'
        ]

    def analyze_token(self, token_name: str, token_symbol: str, token_metadata: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Add more meme keywords
            self.meme_keywords.extend(['moon', 'inu', 'elon', 'pepe', 'rocket'])
            
            has_meme_keyword = any(
                keyword.lower() in token_name.lower() or keyword.lower() in token_symbol.lower()
                for keyword in self.meme_keywords
            )

            score = 0
            if has_meme_keyword:
                score += 40  # Increase from original weight

            # Check supply
            if token_metadata.get('total_supply', 0) > 1_000_000_000_000:
                score += 20

            # Check holder concentration
            top_holders = token_metadata.get('holder_distribution', {}).get('top_holders', {})
            if sum(float(pct) for pct in top_holders.values()) > 50:
                score += 20

            # Check token age
            if self._is_new_token(token_metadata.get('created_at')):
                score += 20

            return {
                'is_meme': score >= 60,
                'meme_score': score,
                'risk_level': 'high' if score >= 80 else 'medium',
                'confidence': min(score, 100)
            }
        except Exception as e:
            logger.error(f"Error analyzing token: {str(e)}")
            return {
                'is_meme': False,
                'meme_score': 0,
                'risk_level': 'unknown',
                'confidence': 0
            }

    def _analyze_meme_characteristics(self, 
                                    name: str, 
                                    symbol: str, 
                                    metadata: Dict) -> Dict[str, bool]:
        factors = {}
        
        factors['has_meme_keyword'] = any(
            keyword in name or keyword in symbol
            for keyword in self.meme_keywords
        )
        
        factors['all_caps_symbol'] = symbol.isupper()
        factors['has_numbers'] = bool(re.search(r'\d', name + symbol))
        factors['special_chars'] = bool(re.search(r'[!@#$%^&*()_+]', name + symbol))
        
        if 'created_at' in metadata:
            factors['is_new_token'] = self._is_new_token(metadata['created_at'])
        
        factors['has_emoji'] = self._contains_emoji(name)
        
        return factors

    def _analyze_risk_factors(self, 
                            name: str, 
                            symbol: str, 
                            metadata: Dict) -> Dict[str, bool]:
        factors = {}
        
        factors['has_risk_keyword'] = any(
            keyword in name or keyword in symbol
            for keyword in self.risk_keywords
        )
        
        if 'total_supply' in metadata:
            factors['suspicious_supply'] = self._is_suspicious_supply(
                metadata['total_supply']
            )
        
        factors['unverified_contract'] = not metadata.get('is_verified', False)
        
        if 'holder_distribution' in metadata:
            factors['concentrated_holdings'] = self._check_holder_concentration(
                metadata['holder_distribution']
            )
            
        return factors

    @staticmethod
    def _is_new_token(created_at: Any) -> bool:
        if isinstance(created_at, (int, float)):
            creation_date = datetime.fromtimestamp(created_at)
            return (datetime.now() - creation_date).days <= 7
        return False

    @staticmethod
    def _contains_emoji(text: str) -> bool:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return bool(emoji_pattern.search(text))

    @staticmethod
    def _is_suspicious_supply(supply: int) -> bool:
        return supply < 1000 or supply > 1_000_000_000_000_000

    @staticmethod
    def _check_holder_concentration(distribution: Dict) -> bool:
        return sum(distribution.get('top_holders', {}).values()) > 80

    def _calculate_meme_score(self, factors: Dict[str, bool]) -> float:
        weights = {
            'has_meme_keyword': 40,
            'all_caps_symbol': 10,
            'has_numbers': 10,
            'special_chars': 10,
            'is_new_token': 15,
            'has_emoji': 15
        }
        
        score = sum(weight for factor, weight in weights.items() 
                   if factors.get(factor, False))
                
        return min(float(score), 100.0)

    def _calculate_risk_score(self, factors: Dict[str, bool]) -> float:
        weights = {
            'has_risk_keyword': 25,
            'suspicious_supply': 25,
            'unverified_contract': 25,
            'concentrated_holdings': 25
        }
        
        score = sum(weight for factor, weight in weights.items() 
                   if factors.get(factor, False))
                
        return min(float(score), 100.0)

    def _determine_risk_level(self, risk_score: float) -> str:
        if risk_score >= 75:
            return 'very_high'
        elif risk_score >= 50:
            return 'high'
        elif risk_score >= 25:
            return 'medium'
        return 'low'

    def _calculate_confidence(self, 
                            meme_factors: Dict[str, bool], 
                            risk_factors: Dict[str, bool]) -> float:
        total_factors = len(meme_factors) + len(risk_factors)
        if total_factors == 0:
            return 0.0
            
        factors_found = sum(1 for v in meme_factors.values() if v) + \
                       sum(1 for v in risk_factors.values() if v)
        
        return float(factors_found / total_factors * 100)
