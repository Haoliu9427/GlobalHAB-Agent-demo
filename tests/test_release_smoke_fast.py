from globalhab_demo.data import generate_demo_data
from globalhab_demo.scenario import project_synthetic_scenario


def test_release_imports_and_core_data_contract():
    frame = generate_demo_data(days=365, seed=42)
    assert len(frame) == 365 * 4
    assert {"date", "region", "sst_c", "mhw_intensity_c", "hab_event"}.issubset(frame.columns)
    assert frame["mhw_intensity_c"].ge(0).all()

    scenario = project_synthetic_scenario(
        issue_date=frame["date"].max().date(),
        horizon_days=14,
        mhw_intensity_c=2.5,
        nitrate_mmol_m3=5.0,
        phosphate_mmol_m3=0.7,
        silicate_mmol_m3=6.5,
        transport_proxy=0.8,
    )
    assert len(scenario) == 12
    assert scenario["综合风险指数"].between(0, 100).all()
