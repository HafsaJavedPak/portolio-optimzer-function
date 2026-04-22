import azure.functions as func
import azure.durable_functions as df
import logging
import json

app = df.DFFunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ─── 1. THE STARTER (Called by your index.html) ───
@app.route(route="run_full_pipeline", methods=["POST"])
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    try:
        payload = req.get_json()
        # Start the background orchestration
        instance_id = await client.start_new("portfolio_orchestrator", None, payload)
        
        logging.info(f"Started orchestration with ID = '{instance_id}'.")
        
        # Returns a 202 Accepted with the 'statusQueryGetUri' for polling
        return client.create_check_status_response(req, instance_id)
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=400)

# ─── 2. THE ORCHESTRATOR (Manages the workflow) ───
@app.orchestration_trigger(context_name="context")
def portfolio_orchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input()
    
    # This calls the heavy function and waits for it without blocking the thread
    result = yield context.call_activity("run_heavy_math_activity", payload)
    return result

# ─── 3. THE ACTIVITY (Your actual AI/Quantum logic) ───
@app.activity_trigger(input_name="payload")
def run_heavy_math_activity(payload: dict):
    # Lazy imports inside the activity to save memory on startup
    from ai_module_code import run_analysis_pipeline
    from quantum_portfolio_optimization import run_quantum_allocation

    tickers_subset = payload.get("tickers")
    method = payload.get("method", "quantum_inspired")
    investment_amount = payload.get("investment_amount", 100000.0)
    risk_aversion = payload.get("risk_aversion", 0.5)

    # 1. Heavy Forecasting (ARIMAX/GARCH)
    pipeline_results = run_analysis_pipeline(tickers_subset)

    # 2. Prep data for Optimizer
    tickers = [r['ticker'] for r in pipeline_results]
    expected_returns = {r['ticker']: r['expected_return'] for r in pipeline_results}
    covariance_matrix = {
        t1: {
            t2: pipeline_results[i]['expected_volatility'] ** 2 if t1 == t2 else 0.0
            for j, t2 in enumerate(tickers)
        }
        for i, t1 in enumerate(tickers)
    }

    optimizer_input = {
        "tickers": tickers,
        "expected_returns": expected_returns,
        "covariance_matrix": covariance_matrix,
        "investment_amount": investment_amount,
        "risk_aversion": risk_aversion
    }

    # 3. Heavy Optimization (Quantum/Classical)
    result = run_quantum_allocation(optimizer_input, method=method)
    result["forecast_inputs"] = pipeline_results
    
    return result
