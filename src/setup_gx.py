import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration


def setup_suite():
    context = gx.get_context()
    suite_name = "order_input_suite"

    try:
        context.delete_expectation_suite(suite_name)
    except Exception:
        pass

    suite = context.add_expectation_suite(suite_name)

    # 1. Timestamps must not be null
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "order_purchase_timestamp"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "order_approved_at"},
        )
    )

    # 2. Numerical ranges (Business logic)
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "item_count", "min_value": 1},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "total_price", "min_value": 0},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "total_freight_value", "min_value": 0},
        )
    )

    # 3. Categorical regex (Brazilian states: 2 uppercase letters)
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={"column": "customer_state", "regex": "^[A-Z]{2}$"},
        )
    )

    context.save_expectation_suite(suite, overwrite=True)
    print(f"✅ Successfully created expectation suite: {suite_name}")


if __name__ == "__main__":
    setup_suite()
