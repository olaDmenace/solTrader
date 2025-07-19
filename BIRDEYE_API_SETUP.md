# 🔑 Birdeye API Setup Guide

## Getting Your API Key

The Birdeye API requires authentication for production use. Here's how to get started:

### 1. Visit Birdeye Website
Go to: https://birdeye.so/

### 2. Create Account / Sign In
- Click "Sign Up" or "Login" 
- Use your email or connect with wallet
- Verify your account

### 3. Access API Dashboard
- Navigate to your profile/dashboard
- Look for "API" or "Developer" section
- Request API access if not immediately available

### 4. Generate API Key
- Create a new API key
- Copy and save it securely
- Note any rate limits for your tier

## Configuration Options

### Option 1: Environment Variable (Recommended)
```bash
export BIRDEYE_API_KEY="your_api_key_here"
```

### Option 2: Add to .env file
```env
BIRDEYE_API_KEY=your_api_key_here
```

### Option 3: Direct in settings.py
```python
BIRDEYE_API_KEY = "your_api_key_here"
```

## Testing with Real API

Once you have your API key:

```bash
# Set your API key
export BIRDEYE_API_KEY="your_api_key_here"

# Run the real API test
python test_birdeye_integration.py
```

## Rate Limits by Tier

| Tier | Requests/Hour | Cost |
|------|---------------|------|
| Free | 100 | $0 |
| Basic | 1,000 | ~$10/month |
| Pro | 10,000 | ~$50/month |
| Enterprise | 100,000+ | Custom |

## API Endpoints Used

Our integration uses these Birdeye endpoints:

- **Trending Tokens**: `/defi/trending_token`
  - Returns top trending tokens by rank
  - Includes price change, volume, market cap
  - Updates every few minutes

## Fallback Options

If you don't have an API key yet, you can still use the integration:

### 1. Run Mock Tests
```bash
python test_birdeye_mock.py
```

### 2. Use Permissive Mode
Set `TRENDING_FALLBACK_MODE=permissive` to continue trading without trending validation when API is unavailable.

### 3. Disable Trending Filter
Set `ENABLE_TRENDING_FILTER=false` to disable trending validation entirely.

## Alternative Data Sources

If Birdeye API is not available, consider these alternatives:

1. **CoinGecko Trending API** - Free tier available
2. **DEX Screener API** - Free with rate limits  
3. **Jupiter Price API** - Free volume/price data
4. **Custom Social Sentiment** - Twitter/Discord mentions

## Production Recommendations

1. **Start with Free Tier** - Test integration thoroughly
2. **Monitor Usage** - Track API calls per hour
3. **Implement Caching** - Reduce API calls (already built-in)
4. **Set Alerts** - Monitor for rate limit hits
5. **Have Fallback** - Graceful degradation when API unavailable

## Troubleshooting

### 401 Unauthorized
- ✅ Check API key is correct
- ✅ Verify API key is not expired
- ✅ Ensure account has API access

### 429 Rate Limited  
- ✅ Check your tier's rate limits
- ✅ Increase request intervals
- ✅ Consider upgrading tier

### 503 Service Unavailable
- ✅ Check Birdeye status page
- ✅ Enable fallback mode
- ✅ Retry with exponential backoff

---

**Ready to enhance your trading with real trending data! 🚀**