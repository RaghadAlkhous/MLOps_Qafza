import great_expectations as gx
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# Load context and suite ONCE at module import time (Singleton)
_gx_context = gx.get_context(context_root_dir="gx")
_gx_suite = _gx_context.get_expectation_suite("order_input_suite")


def validate_order_data(df: pd.DataFrame) -> dict:
    try:
        dataset = gx.from_pandas(df)
        results = dataset.validate(expectation_suite=_gx_suite)

        if results.success:
            return {"success": True, "status": "success", "details": []}

        failed = []
        for result in results.results:
            if not result.success:
                col = result.expectation_config.kwargs.get("column", "unknown")
                exp_type = result.expectation_config.expectation_type
                failed.append(f"Column '{col}' failed '{exp_type}'")

        logger.warning(f"Data validation failed: {failed}")
        return {"success": False, "status": "rejected", "details": failed}

    except Exception as e:
        logger.error(f"Validation engine error: {str(e)}")
        return {
            "success": False,
            "status": "engine_error",
            "details": [f"Validation engine error: {str(e)}"],
        }
