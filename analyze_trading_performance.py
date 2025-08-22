#!/usr/bin/env python3
"""
SolTrader Performance Analysis
Comprehensive analysis of today's trading performance with $10 per trade assumption
"""
import json
from datetime import datetime
from pathlib import Path

def analyze_trading_performance():
    """Analyze trading performance from dashboard data"""
    print("SOLTRADER DAILY PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Investment Model: $10 per trade")
    print()
    
    # Load dashboard data
    try:
        with open("dashboard_data.json", 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    trades = data.get('trades', [])
    
    # Separate closed and open trades
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    open_trades = [t for t in trades if t.get('status') == 'open']
    
    print(f"TRADE SUMMARY")
    print(f"Total Trades Initiated: {len(trades)}")
    print(f"Closed Trades: {len(closed_trades)}")
    print(f"Open Positions: {len(open_trades)}")
    print()
    
    # Analysis of closed trades with $10 per trade
    if not closed_trades:
        print("No closed trades to analyze.")
        return
    
    print(f"PROFIT & LOSS ANALYSIS (Closed Trades Only)")
    print("-" * 45)
    
    total_investment = len(closed_trades) * 10  # $10 per trade
    total_returns = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    
    trade_details = []
    
    for i, trade in enumerate(closed_trades, 1):
        pnl_percentage = trade.get('pnl_percentage', 0)
        
        # Calculate return with $10 investment
        trade_return = 10 + (10 * pnl_percentage / 100)
        profit_loss = trade_return - 10
        
        total_returns += trade_return
        
        if profit_loss > 0:
            winning_trades += 1
            total_profit += profit_loss
        elif profit_loss < 0:
            losing_trades += 1
            total_loss += abs(profit_loss)
        
        trade_details.append({
            'number': i,
            'token': trade.get('token_address', 'Unknown')[:12] + "...",
            'entry_time': trade.get('timestamp', '')[:19],
            'exit_time': trade.get('exit_time', '')[:19],
            'pnl_percentage': pnl_percentage,
            'investment': 10,
            'return': trade_return,
            'profit_loss': profit_loss,
            'reason': trade.get('reason', 'unknown')
        })
    
    net_profit = total_returns - total_investment
    roi_percentage = (net_profit / total_investment) * 100
    win_rate = (winning_trades / len(closed_trades)) * 100
    
    print(f"Total Investment: ${total_investment:,.2f}")
    print(f"Total Returns: ${total_returns:,.2f}")
    print(f"Net Profit/Loss: ${net_profit:+.2f}")
    print(f"ROI: {roi_percentage:+.2f}%")
    print()
    print(f"Win Rate: {win_rate:.1f}% ({winning_trades}/{len(closed_trades)})")
    print(f"Winning Trades: {winning_trades} (${total_profit:+.2f})")
    print(f"Losing Trades: {losing_trades} (${-total_loss:.2f})")
    print()
    
    # Best and worst trades
    if trade_details:
        best_trade = max(trade_details, key=lambda x: x['profit_loss'])
        worst_trade = min(trade_details, key=lambda x: x['profit_loss'])
        
        print(f"BEST TRADE")
        print(f"Token: {best_trade['token']}")
        print(f"Profit: ${best_trade['profit_loss']:+.2f} ({best_trade['pnl_percentage']:+.2f}%)")
        print(f"Return: ${best_trade['return']:.2f} (${best_trade['investment']:.2f} -> ${best_trade['return']:.2f})")
        print()
        
        print(f"WORST TRADE") 
        print(f"Token: {worst_trade['token']}")
        print(f"Loss: ${worst_trade['profit_loss']:+.2f} ({worst_trade['pnl_percentage']:+.2f}%)")
        print(f"Return: ${worst_trade['return']:.2f} (${worst_trade['investment']:.2f} -> ${worst_trade['return']:.2f})")
        print()
    
    # Performance metrics
    profitable_percentages = [t['pnl_percentage'] for t in trade_details if t['profit_loss'] > 0]
    losing_percentages = [t['pnl_percentage'] for t in trade_details if t['profit_loss'] < 0]
    
    print(f"PERFORMANCE METRICS")
    print("-" * 25)
    if profitable_percentages:
        avg_win = sum(t['profit_loss'] for t in trade_details if t['profit_loss'] > 0) / winning_trades
        avg_win_pct = sum(profitable_percentages) / len(profitable_percentages)
        print(f"Average Win: ${avg_win:.2f} ({avg_win_pct:.1f}%)")
    
    if losing_percentages:
        avg_loss = sum(t['profit_loss'] for t in trade_details if t['profit_loss'] < 0) / losing_trades
        avg_loss_pct = sum(losing_percentages) / len(losing_percentages)
        print(f"Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.1f}%)")
    
    if winning_trades > 0 and losing_trades > 0:
        profit_factor = total_profit / total_loss
        print(f"Profit Factor: {profit_factor:.2f}")
    
    # Calculate average holding time
    holding_times = []
    for trade in closed_trades:
        if trade.get('timestamp') and trade.get('exit_time'):
            try:
                entry = datetime.fromisoformat(trade['timestamp'])
                exit = datetime.fromisoformat(trade['exit_time'])
                duration = (exit - entry).total_seconds() / 60  # minutes
                holding_times.append(duration)
            except:
                pass
    
    if holding_times:
        avg_hold_time = sum(holding_times) / len(holding_times)
        print(f"Average Hold Time: {avg_hold_time:.1f} minutes")
    print()
    
    # Show all trades
    print(f"DETAILED TRADE LOG")
    print("-" * 80)
    print(f"{'#':<3} {'Token':<15} {'Entry%':<8} {'Return':<8} {'P&L':<8} {'Exit Reason':<12}")
    print("-" * 80)
    
    for trade in trade_details:
        print(f"{trade['number']:<3} "
              f"{trade['token']:<15} "
              f"{trade['pnl_percentage']:+7.1f}% "
              f"${trade['return']:<7.2f} "
              f"${trade['profit_loss']:+7.2f} "
              f"{trade['reason']:<12}")
    
    print("-" * 80)
    print(f"TOTAL: {len(closed_trades)} trades | "
          f"${total_investment:.2f} invested | "
          f"${total_returns:.2f} returned | "
          f"${net_profit:+.2f} net")
    print()
    
    # Open positions analysis
    if open_trades:
        print(f"OPEN POSITIONS ANALYSIS")
        print("-" * 30)
        print(f"Open Positions: {len(open_trades)}")
        print(f"Capital at Risk: ${len(open_trades) * 10:.2f}")
        print()
        
        print("Current Open Positions:")
        for i, trade in enumerate(open_trades, 1):
            token = trade.get('token_address', 'Unknown')[:12] + "..."
            entry_time = trade.get('timestamp', '')[:19]
            entry_price = trade.get('price', 0)
            print(f"{i}. {token} | Entry: ${entry_price:.6f} | Time: {entry_time}")
    
    print()
    print(f"SUMMARY")
    print("-" * 15)
    
    status = "PROFITABLE" if net_profit > 0 else "LOSING" if net_profit < 0 else "BREAKEVEN"
    
    print(f"Trading Result: {status}")
    print(f"Total Profit/Loss: ${net_profit:+.2f}")
    print(f"Return on Investment: {roi_percentage:+.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    
    if net_profit > 0:
        print(f"Congratulations! You made ${net_profit:.2f} profit today!")
    elif net_profit < 0:
        print(f"Today resulted in a ${abs(net_profit):.2f} loss.")
    else:
        print(f"You broke even today.")
    
    # Performance rating
    if roi_percentage > 50:
        rating = "EXCELLENT"
    elif roi_percentage > 20:
        rating = "VERY GOOD"
    elif roi_percentage > 10:
        rating = "GOOD"
    elif roi_percentage > 0:
        rating = "POSITIVE"
    elif roi_percentage > -10:
        rating = "MINOR LOSS"
    else:
        rating = "SIGNIFICANT LOSS"
    
    print(f"Performance Rating: {rating}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    analyze_trading_performance()