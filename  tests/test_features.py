import numpy as np
import pandas as pd
import pytest

from src import features


def test_assert_no_leakage_passes():
    """Valid features should pass the leakage check without raising."""
    valid_features = ["enquiry_acceleration", "AGE", "NETMONTHLYINCOME"]
    features.assert_no_leakage(valid_features)

def test_assert_no_leakage_fails():
    """Leaky features (like NPA counts) should raise an AssertionError."""
    # num_std is a known banned column (standard asset count)
    leaky_features = ["enquiry_acceleration", "num_std", "AGE"]
    with pytest.raises(AssertionError, match="Label-defining columns leaked"):
        features.assert_no_leakage(leaky_features)

def test_fit_and_transform_basic():
    """Test that fit_params learns medians and transform applies them."""
    # Create mock training data
    df_train = pd.DataFrame({
        "NETMONTHLYINCOME": [1000, 2000, np.nan],
        "Time_With_Curr_Empr": [12, 24, 36],
        "enq_L3m": [1, 2, 0],
        "enq_L6m": [1, 2, 0],
        "enq_L12m": [1, 2, 0],
        "tot_enq": [1, 2, 3],
        "Tot_Active_TL": [1, 2, 3],
        "CC_enq_L6m": [0, 1, 0],
        "PL_enq_L6m": [0, 1, 0],
        "time_since_recent_enq": [10, 20, 30],
        "CC_utilization": [0.5, 0.6, -99999], # Note the sentinel
        "PL_utilization": [0.1, 0.2, 0.3],
        "max_unsec_exposure_inPct": [0.1, 0.2, 0.3],
        "pct_currentBal_all_TL": [0.1, 0.2, 0.3],
        "Unsecured_TL": [1, 1, 1],
        "Total_TL_opened_L6M": [0, 0, 0],
        "Total_TL_opened_L12M": [0, 0, 0],
        "Tot_TL_closed_L12M": [0, 0, 0],
        "Age_Oldest_TL": [10, 20, 30],
        "Age_Newest_TL": [1, 2, 3],
        "CC_TL": [0, 0, 0],
        "PL_TL": [0, 0, 0],
        "Total_TL": [1, 2, 3],
        "AGE": [30, 40, 50],
        "EDUCATION": [1, 2, 3],
        "MARITALSTATUS": ["Married", "Single", "Married"],
        "GENDER": ["M", "F", "M"],
    })

    params = features.fit_params(df_train)
    
    # Assert that missing rates and medians were learned
    assert "medians" in params
    assert params["medians"]["NETMONTHLYINCOME"] == 1500.0  # Median of 1000, 2000
    
    # Assert sentinel demotion for CC_utilization
    # median of [0.5, 0.6] is 0.55
    assert params["medians"]["CC_utilization"] == 0.55

    # Test transform
    df_test = pd.DataFrame({
        "NETMONTHLYINCOME": [np.nan],
        "Time_With_Curr_Empr": [12],
        "enq_L3m": [1],
        "enq_L6m": [1],
        "enq_L12m": [1],
        "tot_enq": [1],
        "Tot_Active_TL": [1],
        "CC_enq_L6m": [0],
        "PL_enq_L6m": [0],
        "time_since_recent_enq": [10],
        "CC_utilization": [-99999],
        "PL_utilization": [0.1],
        "max_unsec_exposure_inPct": [0.1],
        "pct_currentBal_all_TL": [0.1],
        "Unsecured_TL": [1],
        "Total_TL_opened_L6M": [0],
        "Total_TL_opened_L12M": [0],
        "Tot_TL_closed_L12M": [0],
        "Age_Oldest_TL": [10],
        "Age_Newest_TL": [1],
        "CC_TL": [0],
        "PL_TL": [0],
        "Total_TL": [1],
        "AGE": [30],
        "EDUCATION": [1],
        "MARITALSTATUS": ["Married"],
        "GENDER": ["M"],
    })
    
    transformed = features.transform(df_test, params)
    
    # Ensure missing income was imputed with median
    assert transformed["NETMONTHLYINCOME"].iloc[0] == 1500.0
    # Ensure missing CC_utilization was imputed with median
    assert transformed["CC_utilization"].iloc[0] == 0.55
    # Ensure engineered features like enquiry_acceleration were calculated
    # 4*1 + 2*1 + 1*1 = 7
    assert transformed["enquiry_acceleration"].iloc[0] == 7.0