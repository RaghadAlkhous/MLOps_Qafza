import great_expectations as gx
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

def validate_order_data(df: pd.DataFrame) -> dict:
    """
    Validates incoming data against the order_input_suite.
    Returns a dict with 'success' (bool) and 'details' (list of errors).
    """
    suite_name = "order_input_suite"
    
    try:
        # 1. Get the context and the suite object
        context = gx.get_context()
        suite = context.get_expectation_suite(suite_name)
        
        # 2. Create a dataset from the DataFrame
        dataset = gx.from_pandas(df)
        
        # 3. Validate using the suite object (not string)
        results = dataset.validate(expectation_suite=suite)
        
        if results.success:
            return {"success": True, "details": []}
        
        # Extract failed expectations
        failed = []
        for result in results.results:
            if not result.success:
                col = result.expectation_config.kwargs.get("column", "unknown")
                exp_type = result.expectation_config.expectation_type
                failed.append(f"Column '{col}' failed '{exp_type}'")
                
        logger.warning(f"Data validation failed: {failed}")
        return {"success": False, "details": failed}
        
    except Exception as e:
        logger.error(f"Validation engine error: {str(e)}")
        return {"success": False, "details": [f"Validation engine error: {str(e)}"]}
