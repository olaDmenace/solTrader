"""
News and event monitoring module for detecting high-risk market periods
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class NewsEvent:
    source: str
    title: str
    content: str
    timestamp: datetime
    impact_score: float  # 0-1 scale
    relevance_score: float  # 0-1 scale
    sentiment_score: float  # -1 to 1 scale
    tags: List[str]

class NewsMonitor:
    def __init__(self, settings: Any):
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None
        self.recent_events: List[NewsEvent] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self.high_risk_threshold = 0.8
        self.news_sources = [
            "solana-foundation",
            "solana-status",
            "jupiter-updates"
        ]

    async def start_monitoring(self) -> None:
        """Start news monitoring"""
        if self._monitor_task and not self._monitor_task.done():
            return

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(enable_cleanup_closed=True)
        )
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("News monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop news monitoring"""
        try:
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

            logger.info("News monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping news monitoring: {e}")
            # Force close session even if there's an error
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                except:
                    pass
                self.session = None

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                new_events = await self._fetch_news()
                await self._process_events(new_events)
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"News monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def _fetch_news(self) -> List[NewsEvent]:
        """Fetch news from various sources"""
        events = []
        for source in self.news_sources:
            try:
                source_events = await self._fetch_source(source)
                events.extend(source_events)
            except Exception as e:
                logger.error(f"Error fetching from {source}: {str(e)}")
        return events

    async def _fetch_source(self, source: str) -> List[NewsEvent]:
        """Fetch news from a specific source"""
        if not self.session:
            return []

        try:
            # Example endpoints - replace with actual APIs
            endpoints = {
                "solana-foundation": "https://api.solana.foundation/announcements",
                "solana-status": "https://status.solana.com/api/v2/incidents",
                "jupiter-updates": "https://api.jup.ag/announcements"
            }

            if source not in endpoints:
                return []

            async with self.session.get(endpoints[source]) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                return self._parse_events(data, source)

        except Exception as e:
            logger.error(f"Error fetching from {source}: {str(e)}")
            return []

    def _parse_events(self, data: Dict[str, Any], source: str) -> List[NewsEvent]:
        """Parse raw API data into NewsEvent objects"""
        events = []
        try:
            for item in data.get('items', []):
                event = NewsEvent(
                    source=source,
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    timestamp=datetime.fromisoformat(item.get('timestamp', '')),
                    impact_score=self._calculate_impact(item),
                    relevance_score=self._calculate_relevance(item),
                    sentiment_score=self._calculate_sentiment(item),
                    tags=item.get('tags', [])
                )
                events.append(event)
        except Exception as e:
            logger.error(f"Error parsing events: {str(e)}")
        return events

    async def _process_events(self, events: List[NewsEvent]) -> None:
        """Process new events and update risk assessment"""
        for event in events:
            if event.impact_score >= self.high_risk_threshold:
                await self._emit_high_risk_alert(event)
            self.recent_events.append(event)

        # Keep only last 24 hours of events
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_events = [
            e for e in self.recent_events 
            if e.timestamp > cutoff
        ]

    async def _emit_high_risk_alert(self, event: NewsEvent) -> None:
        """Emit alert for high-risk event"""
        if hasattr(self.settings, 'alert_system'):
            await self.settings.alert_system.emit_alert(
                level="warning",
                type="high_risk_event",
                message=f"High-risk event detected: {event.title}",
                data={
                    "source": event.source,
                    "impact_score": event.impact_score,
                    "sentiment_score": event.sentiment_score,
                    "timestamp": event.timestamp.isoformat(),
                    "tags": event.tags
                }
            )

    def _calculate_impact(self, item: Dict[str, Any]) -> float:
        """Calculate event impact score"""
        impact_keywords = {
            'critical': 1.0,
            'outage': 0.9,
            'degraded': 0.7,
            'maintenance': 0.5,
            'update': 0.3
        }
        
        score = 0.0
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        for keyword, weight in impact_keywords.items():
            if keyword in text:
                score = max(score, weight)
                
        return score

    def _calculate_relevance(self, item: Dict[str, Any]) -> float:
        """Calculate event relevance score"""
        relevant_tags = {
            'trading': 1.0,
            'liquidity': 0.9,
            'protocol': 0.8,
            'network': 0.7,
            'maintenance': 0.6
        }
        
        score = 0.0
        tags = [t.lower() for t in item.get('tags', [])]
        
        for tag, weight in relevant_tags.items():
            if tag in tags:
                score = max(score, weight)
                
        return score

    def _calculate_sentiment(self, item: Dict[str, Any]) -> float:
        """Calculate event sentiment score"""
        positive_words = {'upgrade', 'improve', 'fix', 'resolve', 'enhance'}
        negative_words = {'issue', 'problem', 'outage', 'degraded', 'delay'}
        
        text = f"{item.get('title', '')} {item.get('content', '')}".lower().split()
        
        pos_count = sum(1 for word in text if word in positive_words)
        neg_count = sum(1 for word in text if word in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
        return (pos_count - neg_count) / total

    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get current market risk assessment based on news"""
        if not self.recent_events:
            return {
                "risk_level": "low",
                "risk_score": 0.0,
                "high_risk_events": 0,
                "latest_events": []
            }

        # Calculate overall risk metrics
        risk_scores = [e.impact_score for e in self.recent_events]
        avg_risk = sum(risk_scores) / len(risk_scores)
        high_risk_count = sum(1 for e in self.recent_events if e.impact_score >= self.high_risk_threshold)

        return {
            "risk_level": "high" if avg_risk >= self.high_risk_threshold else "medium" if avg_risk >= 0.5 else "low",
            "risk_score": avg_risk,
            "high_risk_events": high_risk_count,
            "latest_events": [{
                "title": e.title,
                "source": e.source,
                "impact_score": e.impact_score,
                "timestamp": e.timestamp.isoformat()
            } for e in sorted(self.recent_events, key=lambda x: x.timestamp, reverse=True)[:5]]
        }