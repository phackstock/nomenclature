import copy
import pytest

import numpy as np
import pandas as pd
from nomenclature.core import process
from nomenclature.definition import DataStructureDefinition
from nomenclature.processor.region import RegionProcessor
from pyam import IAMC_IDX, IamDataFrame, assert_iamframe_equal

from conftest import TEST_DATA_DIR


def test_region_processing_rename():
    # Test **only** the renaming aspect, i.e. 3 things:
    # 1. All native regions **with** a renaming property should be renamed correctly
    # 2. All native regions **without** a renaming property should be passed through
    # 3. All regions which are explicitly named should be dropped
    # Testing strategy:
    # 1. Rename region_a -> region_A
    # 2. Leave region_B untouched
    # 3. Drop region_C

    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["model_a", "scen_a", "region_a", "Primary Energy", "EJ/yr", 1, 2],
                ["model_a", "scen_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
                ["model_a", "scen_a", "region_C", "Primary Energy", "EJ/yr", 5, 6],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )

    exp = copy.deepcopy(test_df)
    exp.filter(region=["region_a", "region_B"], inplace=True)
    exp.rename(region={"region_a": "region_A"}, inplace=True)

    obs = process(
        test_df,
        DataStructureDefinition(TEST_DATA_DIR / "region_processing/dsd"),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing/rename_only"
        ),
    )

    assert_iamframe_equal(obs, exp)


def test_region_processing_empty_raises():
    # Test that an empty result of the region-processing raises
    # see also https://github.com/IAMconsortium/pyam/issues/631

    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["model_a", "scen_a", "region_foo", "Primary Energy", "EJ/yr", 1, 2],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )

    with pytest.raises(ValueError, match="The region aggregation for model model_a"):
        process(
            test_df,
            DataStructureDefinition(TEST_DATA_DIR / "region_processing/dsd"),
            processor=RegionProcessor.from_directory(
                TEST_DATA_DIR / "region_processing/rename_only"
            ),
        )


def test_region_processing_no_mapping(simple_df):
    # Test that a model without a mapping is passed untouched

    exp = copy.deepcopy(simple_df)

    obs = process(
        simple_df,
        DataStructureDefinition(TEST_DATA_DIR / "region_processing/dsd"),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing/no_mapping"
        ),
    )
    assert_iamframe_equal(obs, exp)


def test_region_processing_aggregate():
    # Test only the aggregation feature
    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["model_a", "scen_a", "region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["model_a", "scen_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
                ["model_a", "scen_a", "region_C", "Primary Energy", "EJ/yr", 5, 6],
                ["model_a", "scen_b", "region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["model_a", "scen_b", "region_B", "Primary Energy", "EJ/yr", 3, 4],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )
    exp = IamDataFrame(
        pd.DataFrame(
            [
                ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 4, 6],
                ["model_a", "scen_b", "World", "Primary Energy", "EJ/yr", 4, 6],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )

    obs = process(
        test_df,
        DataStructureDefinition(TEST_DATA_DIR / "region_processing/dsd"),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing/aggregate_only"
        ),
    )

    assert_iamframe_equal(obs, exp)


@pytest.mark.parametrize(
    "directory", ("complete_processing", "complete_processing_list")
)
def test_region_processing_complete(directory):
    # Test all three aspects of region processing together:
    # 1. Renaming
    # 2. Passing models without a mapping
    # 3. Aggregating correctly

    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["m_a", "s_a", "region_a", "Primary Energy", "EJ/yr", 1, 2],
                ["m_a", "s_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
                ["m_a", "s_a", "region_C", "Primary Energy", "EJ/yr", 5, 6],
                ["m_a", "s_a", "region_a", "Primary Energy|Coal", "EJ/yr", 0.5, 1],
                ["m_a", "s_a", "region_B", "Primary Energy|Coal", "EJ/yr", 1.5, 2],
                ["m_b", "s_b", "region_A", "Primary Energy", "EJ/yr", 1, 2],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )
    exp = IamDataFrame(
        pd.DataFrame(
            [
                ["m_a", "s_a", "region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["m_a", "s_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
                ["m_a", "s_a", "World", "Primary Energy", "EJ/yr", 4, 6],
                ["m_a", "s_a", "region_A", "Primary Energy|Coal", "EJ/yr", 0.5, 1],
                ["m_a", "s_a", "region_B", "Primary Energy|Coal", "EJ/yr", 1.5, 2],
                ["m_a", "s_a", "World", "Primary Energy|Coal", "EJ/yr", 2, 3],
                ["m_b", "s_b", "region_A", "Primary Energy", "EJ/yr", 1, 2],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )

    obs = process(
        test_df,
        DataStructureDefinition(TEST_DATA_DIR / "region_processing/dsd"),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing" / directory
        ),
    )
    assert_iamframe_equal(obs, exp)


@pytest.mark.parametrize(
    "folder, exp_df, args",
    [
        (
            "weighted_aggregation",
            [
                ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 4, 6],
                ["model_a", "scen_a", "World", "Emissions|CO2", "Mt CO2", 5, 8],
                ["model_a", "scen_a", "World", "Price|Carbon", "USD/t CO2", 2.8, 7.0],
            ],
            None,
        ),
        (
            "weighted_aggregation_rename",
            [
                ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 4, 6],
                ["model_a", "scen_a", "World", "Emissions|CO2", "Mt CO2", 5, 8],
                ["model_a", "scen_a", "World", "Price|Carbon", "USD/t CO2", 2.8, 7.0],
                ["model_a", "scen_a", "World", "Price|Carbon (Max)", "USD/t CO2", 3, 8],
            ],
            None,
        ),
        # check that region-aggregation with missing weights passes (inconsistent index)
        # TODO check the log output
        (
            "weighted_aggregation",
            [
                ["model_a", "scen_a", "World", "Primary Energy", "EJ/yr", 4, 6],
                ["model_a", "scen_a", "World", "Emissions|CO2", "Mt CO2", 5, np.nan],
            ],
            dict(variable="Emissions|CO2", year=2010, keep=False),
        ),
    ],
)
def test_region_processing_weighted_aggregation(folder, exp_df, args):
    # test a weighed sum

    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["model_a", "scen_a", "region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["model_a", "scen_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
                ["model_a", "scen_a", "region_A", "Emissions|CO2", "Mt CO2", 4, 6],
                ["model_a", "scen_a", "region_B", "Emissions|CO2", "Mt CO2", 1, 2],
                ["model_a", "scen_a", "region_A", "Price|Carbon", "USD/t CO2", 3, 8],
                ["model_a", "scen_a", "region_B", "Price|Carbon", "USD/t CO2", 2, 4],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )

    if args is not None:
        test_df = test_df.filter(**args)

    exp = IamDataFrame(pd.DataFrame(exp_df, columns=IAMC_IDX + [2005, 2010]))

    obs = process(
        test_df,
        DataStructureDefinition(TEST_DATA_DIR / "region_processing" / folder / "dsd"),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing" / folder / "aggregate"
        ),
    )
    assert_iamframe_equal(obs, exp)


def test_region_processing_skip_aggregation():
    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["m_a", "s_a", "region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["m_a", "s_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )
    exp = test_df

    obs = process(
        test_df,
        DataStructureDefinition(
            TEST_DATA_DIR / "region_processing/skip_aggregation/dsd"
        ),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing/skip_aggregation/mappings"
        ),
    )
    assert_iamframe_equal(obs, exp)


def test_partial_aggregation(caplog):
    # Dedicated test for partial aggregation
    # Tests the following two aspects of partial aggregation:
    # 1. A variable that is only found in the common region will be taken from there
    # 2. A variable that is found in both the common region as well as the constituent
    #    regions will be taken from the common region.
    # The two common regions common_region_A and B differ in that common_region_A
    # reports both Primary Energy as well as Temperature|Mean so both values are taken
    # from there. common_region_B, however only reports Temperature|Mean so Primary
    # Energy is obtained through region aggregation adding up regions B and C.

    test_df = IamDataFrame(
        pd.DataFrame(
            [
                ["m_a", "s_a", "region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["m_a", "s_a", "region_B", "Primary Energy", "EJ/yr", 3, 4],
                ["m_a", "s_a", "region_C", "Primary Energy", "EJ/yr", 5, 6],
                ["m_a", "s_a", "common_region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["m_a", "s_a", "common_region_A", "Temperature|Mean", "C", 3, 4],
                ["m_a", "s_a", "common_region_B", "Temperature|Mean", "C", 1, 2],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )
    exp = IamDataFrame(
        pd.DataFrame(
            [
                ["m_a", "s_a", "common_region_A", "Primary Energy", "EJ/yr", 1, 2],
                ["m_a", "s_a", "common_region_A", "Temperature|Mean", "C", 3, 4],
                ["m_a", "s_a", "common_region_B", "Primary Energy", "EJ/yr", 8, 10],
                ["m_a", "s_a", "common_region_B", "Temperature|Mean", "C", 1, 2],
            ],
            columns=IAMC_IDX + [2005, 2010],
        )
    )

    obs = process(
        test_df,
        DataStructureDefinition(TEST_DATA_DIR / "region_processing/dsd"),
        processor=RegionProcessor.from_directory(
            TEST_DATA_DIR / "region_processing/partial_aggregation"
        ),
    )
    # Assert that we get the expected values
    assert_iamframe_equal(obs, exp)
    # Assert that we the the appropriate warnings since there is a mismatch between
    # in Primary Energy between model native and aggregated values for common_region_A
    log_content = [
        "Differences found between model native and aggregated results",
        "m_a",
        "s_a",
        "common_region_A",
        "Primary Energy",
        "EJ/yr",
        "2005             1           4",
        "2010             2           6",
    ]

    assert all(c in caplog.text for c in log_content)
