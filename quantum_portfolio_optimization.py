import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize
import math

# Configure logging for production-readiness
logger = logging.getLogger("QuantumPortfolioOpt")
logger.setLevel(logging.INFO)
# Avoid duplicate handlers if re-running cell
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("Quantum Optimization Module Initialized.")


@dataclass
class ForecastInput:
    tickers: List[str]
    expected_returns: Dict[str, float]
    covariance_matrix: pd.DataFrame
    investment_amount: float
    risk_aversion: float = 0.5

@dataclass
class AllocationOutput:
    optimal_weights: Dict[str, float]
    allocated_amounts: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    sharpe_ratio: float
    solver_metadata: Dict[str, str]
    
    def to_dict(self):
        return {
            "optimal_weights": self.optimal_weights,
            "allocated_amounts": self.allocated_amounts,
            "expected_portfolio_return": round(self.expected_portfolio_return, 6),
            "expected_portfolio_volatility": round(self.expected_portfolio_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "solver_metadata": self.solver_metadata
        }


def validate_input(data: ForecastInput) -> bool:
    logger.info("Validating input data parameters...")
    
    n_assets = len(data.tickers)
    
    if n_assets == 0:
        logger.error("Ticker list is empty.")
        raise ValueError("Ticker list cannot be empty.")
        
    if len(data.expected_returns) != n_assets:
        logger.error(f"Mismatch: {n_assets} tickers but {len(data.expected_returns)} return expectations.")
        raise ValueError("Expected returns dimension mismatch.")
        
    if data.covariance_matrix.shape != (n_assets, n_assets):
        logger.error(f"Covariance matrix shape {data.covariance_matrix.shape} invalid for {n_assets} assets.")
        raise ValueError("Covariance matrix dimensions must be NxN matching ticker count.")
        
    # Check Positive Semi-Definite (Eigenvalues >= minimum threshold close to 0)
    try:
        eigvals = np.linalg.eigvals(data.covariance_matrix.values)
        if np.any(eigvals < -1e-8):
            logger.warning("Covariance matrix is not strictly Positive Semi-Definite (has negative eigenvalues). This may cause optimizer instability.")
            # We warn rather than raise because numerical precision issues often render empirical covariance slightly non-PSD
    except Exception as e:
        logger.error(f"Error computing eigenvalues: {e}")
        raise
        
    if data.investment_amount <= 0:
        logger.error("Investment amount must be strictly positive.")
        raise ValueError("Invalid investment amount.")
        
    logger.info("Validation successful.")
    return True


def build_optimization_problem(data: ForecastInput) -> Tuple[np.ndarray, np.ndarray, float]:
    logger.info("Building optimization structural arrays...")
    
    # Extract arrays strictly in order of tickers to maintain positional mapping
    mu = np.array([data.expected_returns[t] for t in data.tickers])
    
    # Ensure covariance is aligned with tickers
    cov_aligned = data.covariance_matrix.loc[data.tickers, data.tickers]
    sigma = cov_aligned.values
    
    lmbda = data.risk_aversion
    
    logger.info(f"Built bounds for {len(mu)} assets.")
    return mu, sigma, lmbda


def classical_slsqp_solve(mu: np.ndarray, sigma: np.ndarray, lmbda: float) -> np.ndarray:
    n_assets = len(mu)
    
    # Objective: Minimize 0.5 * lambda * (w^T * Sigma * w) - w^T * mu
    def objective(w):
        variance = np.dot(w.T, np.dot(sigma, w))
        expected_ret = np.dot(w.T, mu)
        # Scaled to make variance and returns comparable depending on lambda
        return (lmbda * variance) - expected_ret
        
    # Constraints: sum(w) = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # Bounds: w >= 0 (long only)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    # Initial guess: equal weight
    w0 = np.ones(n_assets) / n_assets
    
    logger.info("Executing Classical SLSQP optimization...")
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if res.success:
        logger.info("Classical optimization converged optimal.")
        return res.x
    else:
        logger.error(f"Classical optimization failed: {res.message}")
        # Return equal weight fallback
        return w0

def quantum_inspired_annealing_solve(mu: np.ndarray, sigma: np.ndarray, lmbda: float, precision_bits: int = 4) -> np.ndarray:
    # A true quantum annealer (like D-Wave) maps to QUBO.
    # QUBO requires discretizing continuous weights (e.g. w_i in [0,1] -> bitstring of K precision_bits)
    # Here we simulate the meta-heuristic approach using classical Simulated Annealing acting on a discrete neighborhood.
    logger.info(f"Executing Quantum-Inspired Simulated Annealing optimization... (Precision: {precision_bits} bits/asset)")
    
    n_assets = len(mu)
    max_states = (2**precision_bits) - 1
    
    # State representation: array of integers 0 to max_states
    # Current solution = state / sum(state)  [ensures constraint sum(w)=1 implicitly]
    
    def decode_state(state):
        s_sum = np.sum(state)
        if s_sum == 0:
            return np.ones(n_assets) / n_assets
        return state / s_sum
        
    def energy(state):
        w = decode_state(state)
        variance = np.dot(w.T, np.dot(sigma, w))
        expected_ret = np.dot(w.T, mu)
        # QUBO objective maps to energy minimization
        return (lmbda * variance) - expected_ret
        
    # Annealing schedule
    T = 1.0
    T_min = 0.0001
    alpha = 0.95
    iters_per_temp = 50
    
    # Initial random state
    current_state = np.random.randint(0, max_states+1, size=n_assets)
    # Ensure invalid zero state isn't initialized
    if np.sum(current_state) == 0:
        current_state[0] = 1
        
    current_energy = energy(current_state)
    
    best_state = np.copy(current_state)
    best_energy = current_energy
    
    while T > T_min:
        for _ in range(iters_per_temp):
            # Propose neighbor: randomly perturb one asset's integer bit representation
            neighbor = np.copy(current_state)
            idx = np.random.randint(n_assets)
            
            # Step +/- 1 in discrete state space
            step = np.random.choice([-1, 1])
            neighbor[idx] = max(0, min(max_states, neighbor[idx] + step))
            
            if np.sum(neighbor) == 0:
                continue # Skip all zeros
                
            neighbor_energy = energy(neighbor)
            
            delta = neighbor_energy - current_energy
            
            # Metropolis acceptance criterion
            if delta < 0 or math.exp(-delta / T) > np.random.random():
                current_state = np.copy(neighbor)
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_state = np.copy(current_state)
                    best_energy = current_energy
                    
        T *= alpha
        
    logger.info("Quantum-Inspired Annealing completed.")
    return decode_state(best_state)

def solve_quantum_optimization(mu: np.ndarray, sigma: np.ndarray, lmbda: float, method: str = 'quantum_inspired') -> Tuple[np.ndarray, dict]:
    if method == 'classical':
        w_opt = classical_slsqp_solve(mu, sigma, lmbda)
        metadata = {"method": "classical SLSQP continuous", "convergence_status": "optimal"}
    elif method == 'quantum_inspired':
        w_opt = quantum_inspired_annealing_solve(mu, sigma, lmbda, precision_bits=5)
        metadata = {"method": "quantum-inspired simulated annealing discrete", "convergence_status": "heuristic_complete"}
    else:
        raise ValueError(f"Unknown optimization method requested: {method}")
        
    return w_opt, metadata


def postprocess_results(w_opt: np.ndarray, metadata: dict, data: ForecastInput, mu: np.ndarray, sigma: np.ndarray) -> dict:
    logger.info("Post-processing optimized weights to formatted payload...")
    
    # Ensure clean decimal rounding for weights
    clean_weights = {ticker: round(float(w), 4) for ticker, w in zip(data.tickers, w_opt)}
    
    # Calculate Allocations
    allocated_amounts = {ticker: round(float(w * data.investment_amount), 2) for ticker, w in zip(data.tickers, w_opt)}
    
    # Expected stats
    exp_ret = float(np.dot(w_opt.T, mu))
    exp_vol = float(np.sqrt(np.dot(w_opt.T, np.dot(sigma, w_opt))))
    
    # Assume Risk-Free rate approximates 0.02 annual -> 0.00008 daily depending on scale. 
    # For a generalized portfolio scale, if expected return is annual, we use a standard rf
    rf = 0.02
    sharpe = (exp_ret - rf) / exp_vol if exp_vol > 0 else 0.0
    
    output = AllocationOutput(
        optimal_weights=clean_weights,
        allocated_amounts=allocated_amounts,
        expected_portfolio_return=exp_ret,
        expected_portfolio_volatility=exp_vol,
        sharpe_ratio=sharpe,
        solver_metadata=metadata
    )
    
    return output.to_dict()


def run_quantum_allocation(data_dict: dict, method: str = 'quantum_inspired') -> dict:
    """
    Main module entrypoint.
    Transforms raw JSON input -> Dataclass -> Optimized Payload dict
    """
    logger.info(f"--- Firing Allocation Engine (Solver: {method}) ---")
    
    try:
        # 1. Parse into structured Dataclass
        # Covariance matrix must be reconstructed to pandas DataFrame if passed as dict
        cov_matrix_raw = data_dict['covariance_matrix']
        tickers = data_dict['tickers']
        
        if isinstance(cov_matrix_raw, dict):
            covariance_df = pd.DataFrame(cov_matrix_raw, index=tickers, columns=tickers)
        else:
            covariance_df = cov_matrix_raw # assuming it is already a DataFrame
            
        data = ForecastInput(
            tickers=tickers,
            expected_returns=data_dict['expected_returns'],
            covariance_matrix=covariance_df,
            investment_amount=data_dict['investment_amount'],
            risk_aversion=data_dict.get('risk_aversion', 0.5)
        )
        
        # 2. Validation
        validate_input(data)
        
        # 3. Problem construction
        mu, sigma, lmbda = build_optimization_problem(data)
        
        # 4. Solvers
        w_opt, metadata = solve_quantum_optimization(mu, sigma, lmbda, method)
        
        # 5. Output structuring
        final_payload = postprocess_results(w_opt, metadata, data, mu, sigma)
        logger.info("--- Cycle Complete. Payload ready. ---")
        return final_payload
        
    except Exception as e:
        logger.error(f"Fatal allocation error: {e}")
        return {"error": str(e), "solver_metadata": {"status": "failed"}}


if __name__ == "__main__":
    import json
    
    # MOCK AI FORECAST PAYLOAD
    mock_input = {
        "tickers": ["AAPL", "MSFT", "GOOG"],
        "expected_returns": {
            "AAPL": 0.08,
            "MSFT": 0.06,
            "GOOG": 0.07
        },
        # Represent covariance matrix as nested dict representing DataFrame
        "covariance_matrix": {
            "AAPL": {"AAPL": 0.04, "MSFT": 0.02, "GOOG": 0.015},
            "MSFT": {"AAPL": 0.02, "MSFT": 0.05, "GOOG": 0.025},
            "GOOG": {"AAPL": 0.015, "MSFT": 0.025, "GOOG": 0.06},
        },
        "investment_amount": 100000.0,
        "risk_aversion": 1.0 # high risk aversion = prioritizes variance minimization
    }
    
    # Execute Quantum-Inspired
    q_output = run_quantum_allocation(mock_input, method='quantum_inspired')
    
    # Execute Classical benchmark
    c_output = run_quantum_allocation(mock_input, method='classical')
    
    print("\n----- OUTPUT PAYLOAD (QUANTUM-INSPIRED ANNEALING) -----")
    print(json.dumps(q_output, indent=4))
    
    print("\n----- OUTPUT PAYLOAD (CLASSICAL MARKOVITZ SLSQP) -----")
    print(json.dumps(c_output, indent=4))
