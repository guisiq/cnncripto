"""
Backtesting and evaluation module
"""
import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

class SimpleBacktester:
    """Simple vectorized backtest of signals"""
    
    def __init__(self, initial_cash: float = 10000.0, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission
    
    def backtest(
        self,
        close_prices: np.ndarray,
        signals: np.ndarray,
        signal_threshold: float = 0.5,
        trade_size: float = 1.0
    ) -> dict:
        """
        Backtest strategy
        
        Args:
            close_prices: (n_candles,) array of close prices
            signals: (n_candles,) array of signal scores in [-1, 1]
            signal_threshold: Threshold for signal generation
            trade_size: Fraction of portfolio to trade (0 to 1)
        
        Returns:
            Dictionary with backtest metrics
        """
        n = len(close_prices)
        
        # Generate buy/sell decisions
        decisions = np.zeros(n)  # 1=buy, -1=sell, 0=hold
        decisions[signals > signal_threshold] = 1
        decisions[signals < -signal_threshold] = -1
        
        # Portfolio tracking
        position = 0  # 1=long, -1=short, 0=flat
        cash = self.initial_cash
        shares = 0
        equity_curve = []
        trades = []
        
        for i in range(n):
            if decisions[i] == 1 and position == 0:  # Open long
                # Buy
                cost = cash * trade_size / (close_prices[i] * (1 + self.commission))
                shares = cost
                cash -= shares * close_prices[i] * (1 + self.commission)
                position = 1
                trades.append(('BUY', close_prices[i], shares))
            
            elif decisions[i] == -1 and position == 0:  # Open short
                # Sell
                # For simplicity, we simulate selling an amount
                shares = (cash * trade_size) / (close_prices[i] * (1 + self.commission))
                cash += shares * close_prices[i] * (1 - self.commission)
                position = -1
                trades.append(('SELL', close_prices[i], shares))
            
            elif decisions[i] == 1 and position == -1:  # Close short, open long
                # Cover short
                cash += shares * close_prices[i] * (1 - self.commission)
                shares = 0
                # Buy
                cost = cash * trade_size / (close_prices[i] * (1 + self.commission))
                shares = cost
                cash -= shares * close_prices[i] * (1 + self.commission)
                position = 1
                trades.append(('CLOSE_SHORT', close_prices[i], 0))
                trades.append(('BUY', close_prices[i], shares))
            
            elif decisions[i] == -1 and position == 1:  # Close long, open short
                # Sell
                cash += shares * close_prices[i] * (1 - self.commission)
                shares = 0
                # Short
                shares = (cash * trade_size) / (close_prices[i] * (1 + self.commission))
                cash += shares * close_prices[i] * (1 - self.commission)
                position = -1
                trades.append(('SELL', close_prices[i], 0))
                trades.append(('SHORT', close_prices[i], shares))
            
            # Calculate equity
            if position == 1:
                equity = cash + shares * close_prices[i]
            elif position == -1:
                equity = cash - shares * close_prices[i]
            else:
                equity = cash
            
            equity_curve.append(equity)
        
        equity_curve = np.array(equity_curve)
        
        # Calculate metrics
        total_return = (equity_curve[-1] - self.initial_cash) / self.initial_cash
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        daily_returns = returns
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe = (np.mean(daily_returns) * 252) / (annual_volatility + 1e-8)
        
        # Max drawdown
        cumsum = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        trade_returns = []
        if len(trades) >= 2:
            for i in range(0, len(trades) - 1, 2):
                entry_price = trades[i][1]
                exit_price = trades[i + 1][1] if i + 1 < len(trades) else close_prices[-1]
                
                if trades[i][0] == 'BUY':
                    tr = (exit_price - entry_price) / entry_price
                else:
                    tr = (entry_price - exit_price) / entry_price
                
                trade_returns.append(tr)
        
        win_rate = len([r for r in trade_returns if r > 0]) / (len(trade_returns) + 1e-8)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'final_equity': equity_curve[-1],
            'equity_curve': equity_curve,
            'returns': returns,
        }
        
        logger.info(
            "backtest_complete",
            total_return=total_return,
            sharpe=sharpe,
            max_dd=max_drawdown,
            trades=len(trades)
        )
        
        return metrics

def calculate_metrics(
    equity_curve: np.ndarray,
    initial_capital: float = 10000.0,
    annual_factor: int = 252
) -> dict:
    """Calculate financial metrics from equity curve"""
    
    if len(equity_curve) < 2:
        return {}
    
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Sharpe
    annual_return = np.mean(daily_returns) * annual_factor
    annual_volatility = np.std(daily_returns) * np.sqrt(annual_factor)
    sharpe = annual_return / (annual_volatility + 1e-8)
    
    # Sortino (downside volatility)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = np.std(downside_returns) * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
    sortino = annual_return / (downside_vol + 1e-8)
    
    # Max Drawdown
    cumsum = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = (cumsum - running_max) / running_max
    max_dd = np.min(drawdown)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
    }

if __name__ == "__main__":
    # Example usage
    backtester = SimpleBacktester()
    
    # Generate dummy price and signal data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))
    signals = np.sin(np.linspace(0, 10*np.pi, 500)) * 0.5
    
    results = backtester.backtest(prices, signals, signal_threshold=0.3)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
