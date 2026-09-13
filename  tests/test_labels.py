import pandas as pd

from src import labels


def test_build_stress_label():
    """Test that the 12-month fixed window stress label is correctly computed."""
    # Create mock dataframe with various delinquency features
    df = pd.DataFrame({
        "num_std_12mts": [1, 0, 0, 0], # Sub-standard asset (not NPA)
        "num_sub_12mts": [0, 1, 0, 0], # Doubtful asset (NPA)
        "num_dbt_12mts": [0, 0, 0, 0],
        "num_lss_12mts": [0, 0, 0, 0],
        "num_deliq_12mts": [0, 0, 2, 0], # 2 delinquencies
        "num_times_30p_dpd": [0, 0, 0, 0], # Over 30 days past due (should not matter if not in 12m)
    })
    
    y = labels.build_stress_label(df)
    
    # Second and third should be stressed
    assert y.iloc[0] == 0 # std is not NPA
    assert y.iloc[1] == 1 # sub is NPA
    assert y.iloc[2] == 1 # deliq > 0
    assert y.iloc[3] == 0 # healthy

def test_exposure_band():
    """Test that trade line counts correctly map to exposure bands."""
    df = pd.DataFrame({
        "Total_TL": [1, 2, 3, 5, 6, 10, 11, 20]
    })
    
    bands = labels.exposure_band(df)
    
    # Verify the mappings
    assert bands.iloc[0] == "1-2"
    assert bands.iloc[1] == "1-2"
    assert bands.iloc[2] == "3-5"
    assert bands.iloc[3] == "3-5"
    assert bands.iloc[4] == "6-10"
    assert bands.iloc[5] == "6-10"
    assert bands.iloc[6] == "11+"
    assert bands.iloc[7] == "11+"