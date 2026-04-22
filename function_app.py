import azure.functions as func
import logging
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Create the FunctionApp instance early but avoid importing heavy modules at module import time.
# app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="optimize_portfolio", methods=["POST"])
def optimize_portfolio(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing portfolio optimization request.')
    try:
        # Lazy import to avoid heavy work during Functions host startup
        from quantum_portfolio_optimization import run_quantum_allocation

        payload = req.get_json()
        method = payload.get("method", "quantum_inspired")
        result = run_quantum_allocation(payload, method=method)
        return func.HttpResponse(json.dumps(result), mimetype="application/json")
    except Exception as e:
        logging.exception("optimize_portfolio handler error")
        return func.HttpResponse(str(e), status_code=400)


@app.route(route="run_full_pipeline", methods=["POST"])
def run_full_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    # 1. Handle the invisible browser 'Handshake'
    # if req.method == "OPTIONS":
    #     return func.HttpResponse(
    #         status_code=204,
    #         headers={
    #             "Access-Control-Allow-Origin": "https://portfoliostoragehafsa.z22.web.core.windows.net",
    #             "Access-Control-Allow-Methods": "POST, OPTIONS",
    #             "Access-Control-Allow-Headers": "Content-Type",
    #         }
    #     )
    
    
    logging.info("Full pipeline triggered via HTTP.")
    try:
        # Lazy imports to keep module import lightweight for the Functions host
        from ai_module_code import run_analysis_pipeline
        from quantum_portfolio_optimization import run_quantum_allocation
        from azure.storage.blob import BlobServiceClient

        payload = req.get_json()
        tickers_subset = payload.get("tickers", None)
        method = payload.get("method", "quantum_inspired")
        investment_amount = payload.get("investment_amount", 100000.0)
        risk_aversion = payload.get("risk_aversion", 0.5)

        pipeline_results = run_analysis_pipeline(tickers_subset)

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

        result = run_quantum_allocation(optimizer_input, method=method)
        result["forecast_inputs"] = pipeline_results

        return func.HttpResponse(
            json.dumps(result, indent=2),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.exception("run_full_pipeline handler error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

# @app.timer_trigger(schedule="0 0 */1 * * *", arg_name="mytimer", run_on_startup=False)
"""
@app.timer_trigger(schedule="0 */3 * * * *", arg_name="mytimer", run_on_startup=False)
def stock_pipeline(mytimer: func.TimerRequest) -> None:
    logging.info("Stock pipeline started.")

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    client = BlobServiceClient.from_connection_string(conn_str)

    # Step 1: Run forecasting pipeline
    pipeline_results = run_analysis_pipeline()
    logging.info(f"Pipeline returned {len(pipeline_results)} tickers.")

    # Step 2: Build optimizer input from pipeline output
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
        "investment_amount": 100000.0,
        "risk_aversion": 0.5
    }

    # Step 3: Run both optimizers
    result_quantum   = run_quantum_allocation(optimizer_input, method="quantum_inspired")
    result_classical = run_quantum_allocation(optimizer_input, method="classical")

    # Step 4: Save to Blob Storage
    output = {
        "date": datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d"),
        "quantum":   result_quantum,
        "classical": result_classical
    }

    # logging.info(output)
    logging.info("Saving results to Blob Storage...")

    blob = client.get_blob_client(
        container="results",
        blob=f"{datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y-%m-%d')}.json"
    )
    blob.upload_blob(json.dumps(output, indent=2), overwrite=True)
    logging.info("Pipeline complete. Result saved.")

"""
