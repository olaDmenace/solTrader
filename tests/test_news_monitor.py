"""
test_news_monitor.py - Tests for the news and event monitoring module
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from src.trading.news_monitor import NewsMonitor, NewsEvent

@pytest.fixture
def mock_settings():
    settings = Mock()
    settings.alert_system = AsyncMock()
    return settings

@pytest.fixture
def sample_news_data():
    return {
        'items': [
            {
                'title': 'Critical Network Upgrade',
                'content': 'Scheduled maintenance upgrade to improve network performance',
                'timestamp': datetime.now().isoformat(),
                'tags': ['network', 'maintenance', 'upgrade']
            },
            {
                'title': 'Trading Volume Update',
                'content': 'New record trading volume achieved on the protocol',
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'tags': ['trading', 'volume', 'milestone']
            },
            {
                'title': 'Network Degradation',
                'content': 'Users experiencing delayed transaction confirmations',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'tags': ['network', 'issue', 'degraded']
            }
        ]
    }

@pytest_asyncio.fixture
async def news_monitor(mock_settings):
    monitor = NewsMonitor(mock_settings)
    yield monitor
    await monitor.stop_monitoring()

@pytest.mark.asyncio
async def test_monitor_initialization(news_monitor):
    """Test monitor initialization"""
    assert news_monitor.settings is not None
    assert news_monitor.session is None
    assert len(news_monitor.recent_events) == 0
    assert news_monitor._monitor_task is None

@pytest.mark.asyncio
async def test_start_stop_monitoring(news_monitor):
    """Test starting and stopping the monitor"""
    await news_monitor.start_monitoring()
    assert news_monitor.session is not None
    assert news_monitor._monitor_task is not None
    
    await news_monitor.stop_monitoring()
    assert news_monitor.session is None
    assert news_monitor._monitor_task.cancelled()

@pytest.mark.asyncio
async def test_fetch_news(news_monitor, sample_news_data):
    """Test news fetching functionality"""
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = sample_news_data
        mock_get.return_value.__aenter__.return_value = mock_response
        
        await news_monitor.start_monitoring()
        events = await news_monitor._fetch_news()
        
        assert len(events) > 0
        assert all(isinstance(e, NewsEvent) for e in events)
        
        # Check event parsing
        event = events[0]
        assert isinstance(event.timestamp, datetime)
        assert 0 <= event.impact_score <= 1
        assert 0 <= event.relevance_score <= 1
        assert -1 <= event.sentiment_score <= 1

@pytest.mark.asyncio
async def test_event_processing(news_monitor):
    """Test event processing and risk assessment"""
    # Create test events
    events = [
        NewsEvent(
            source="test",
            title="High Risk Event",
            content="Critical network issue", 
            timestamp=datetime.now(),
            impact_score=0.9,
            relevance_score=0.8,
            sentiment_score=-0.7,
            tags=["network", "critical"]
        ),
        NewsEvent(
            source="test",
            title="Low Risk Event",
            content="Minor update",
            timestamp=datetime.now(),
            impact_score=0.3,
            relevance_score=0.5,
            sentiment_score=0.2,
            tags=["update"]
        )
    ]

    await news_monitor._process_events(events)

    # Check risk assessment
    assessment = news_monitor.get_risk_assessment()
    assert isinstance(assessment, dict)
    assert "risk_level" in assessment
    assert "risk_score" in assessment
    assert "high_risk_events" in assessment  # Using the actual key name
    assert "latest_events" in assessment
    assert assessment["high_risk_events"] == 1
    assert assessment["risk_level"] in ["low", "medium", "high"]
    assert 0 <= assessment["risk_score"] <= 1

@pytest.mark.asyncio
async def test_impact_calculation(news_monitor):
    """Test impact score calculation"""
    test_items = [
        {
            'title': 'Critical Network Issue',
            'content': 'Major outage reported',
            'tags': ['critical']
        },
        {
            'title': 'Minor Update',
            'content': 'Regular maintenance',
            'tags': ['update']
        }
    ]
    
    scores = [news_monitor._calculate_impact(item) for item in test_items]
    assert scores[0] > scores[1]  # Critical issue should have higher impact
    assert all(0 <= score <= 1 for score in scores)

@pytest.mark.asyncio
async def test_relevance_calculation(news_monitor):
    """Test relevance score calculation"""
    test_items = [
        {
            'title': 'Trading Update',
            'content': 'New trading pairs added',
            'tags': ['trading', 'protocol']
        },
        {
            'title': 'Community Update',
            'content': 'New discord channel',
            'tags': ['community']
        }
    ]
    
    scores = [news_monitor._calculate_relevance(item) for item in test_items]
    assert scores[0] > scores[1]  # Trading update should be more relevant
    assert all(0 <= score <= 1 for score in scores)

@pytest.mark.asyncio
async def test_sentiment_calculation(news_monitor):
    """Test sentiment score calculation"""
    test_items = [
        {
            'title': 'Network Upgrade Success',
            'content': 'Performance improvements implemented',
            'tags': ['upgrade']
        },
        {
            'title': 'Network Issues',
            'content': 'Multiple problems reported',
            'tags': ['issues']
        }
    ]
    
    scores = [news_monitor._calculate_sentiment(item) for item in test_items]
    assert scores[0] > scores[1]  # Positive news should have higher sentiment
    assert all(-1 <= score <= 1 for score in scores)

@pytest.mark.asyncio
async def test_high_risk_alerts(news_monitor, mock_settings):
    """Test high-risk event alerts"""
    high_risk_event = NewsEvent(
        source="test",
        title="Critical Issue",
        content="Major network outage",
        timestamp=datetime.now(),
        impact_score=0.9,
        relevance_score=0.9,
        sentiment_score=-0.8,
        tags=["critical", "network"]
    )
    
    await news_monitor._process_events([high_risk_event])
    
    # Verify alert was emitted
    mock_settings.alert_system.emit_alert.assert_called_once()
    call_args = mock_settings.alert_system.emit_alert.call_args[1]
    assert call_args['level'] == "warning"
    assert call_args['type'] == "high_risk_event"

@pytest.mark.asyncio
async def test_event_cleanup(news_monitor):
    """Test cleanup of old events"""
    old_event = NewsEvent(
        source="test",
        title="Old Event",
        content="Old content",
        timestamp=datetime.now() - timedelta(hours=25),
        impact_score=0.5,
        relevance_score=0.5,
        sentiment_score=0.0,
        tags=[]
    )
    
    new_event = NewsEvent(
        source="test",
        title="New Event",
        content="New content",
        timestamp=datetime.now(),
        impact_score=0.5,
        relevance_score=0.5,
        sentiment_score=0.0,
        tags=[]
    )
    
    news_monitor.recent_events = [old_event, new_event]
    await news_monitor._process_events([])
    
    assert len(news_monitor.recent_events) == 1
    assert news_monitor.recent_events[0].title == "New Event"

@pytest.mark.asyncio
async def test_error_handling(news_monitor):
    """Test error handling during news fetching"""
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Test connection error
        mock_get.side_effect = aiohttp.ClientError()
        events = await news_monitor._fetch_source("test-source")
        assert len(events) == 0
        
        # Test invalid response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_get.side_effect = None
        mock_get.return_value.__aenter__.return_value = mock_response
        events = await news_monitor._fetch_source("test-source")
        assert len(events) == 0

@pytest.mark.asyncio
async def test_concurrent_monitoring(news_monitor, sample_news_data):
    """Test concurrent monitoring operations"""
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = sample_news_data
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Start monitoring multiple times
        await news_monitor.start_monitoring()
        await news_monitor.start_monitoring()  # Should not create duplicate tasks
        
        assert news_monitor._monitor_task is not None
        assert not news_monitor._monitor_task.done()
        
        # Verify only one task is running
        initial_task = news_monitor._monitor_task
        await news_monitor.start_monitoring()
        assert news_monitor._monitor_task == initial_task