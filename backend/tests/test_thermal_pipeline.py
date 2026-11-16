from app.pipeline import run_thermal_inference


def test_thermal_metrics_attribute_anomaly_to_sensor():
    records = [
        {"gearbox_oil_temp": 20.0, "wind_speed": 8.0}
        for _ in range(19)
    ]
    records.append({"gearbox_oil_temp": 100.0, "wind_speed": 8.0})

    result = run_thermal_inference("DZH1", records)
    sensor = result["thermal_metrics"]["gearbox_oil_temp"]

    assert sensor["is_anomaly"] is True
    assert sensor["warning"] == 80
    assert sensor["critical"] == 95
    assert sensor["rate_of_change"] > 1
    assert any(alert["sensor"] == "gearbox_oil_temp" for alert in result["alerts"])


def test_thermal_trend_is_unavailable_with_insufficient_history():
    records = [
        {"gearbox_oil_temp": 30.0, "wind_speed": 8.0}
        for _ in range(5)
    ]

    result = run_thermal_inference("DZH1", records)

    assert result["thermal_metrics"]["gearbox_oil_temp"]["rate_of_change"] is None
