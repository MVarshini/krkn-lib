"""Scenario telemetry payloads for GPU chaos scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from krkn_lib.gpu.models import (
    Availability,
    Evidence,
    GpuSnapshot,
    JsonRecord,
    Measurement,
)


@dataclass
class GpuInjectorResult(JsonRecord):
    """Primitive, allowlisted evidence returned by one GPU injector device."""

    device_uuid: str
    metrics: dict[str, int | float | bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuInjectorResult:
        return cls(
            device_uuid=value["device_uuid"],
            metrics=dict(value.get("metrics") or {}),
        )


@dataclass
class GpuWorkerTelemetry(JsonRecord):
    """GPU injector outcome for a worker node."""

    node_name: str
    phase: str | None = None
    pod_deleted: bool | None = None
    devices: list[GpuInjectorResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuWorkerTelemetry:
        return cls(
            node_name=value["node_name"],
            phase=value.get("phase"),
            pod_deleted=value.get("pod_deleted"),
            devices=[
                item
                if isinstance(item, GpuInjectorResult)
                else GpuInjectorResult.from_dict(item)
                for item in (value.get("devices") or [])
            ],
        )


@dataclass
class GpuObservation(JsonRecord):
    """A GPU metrics snapshot captured for a node and scenario phase."""

    node_name: str
    phase: str
    snapshot: GpuSnapshot

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuObservation:
        snapshot = value["snapshot"]
        if isinstance(snapshot, dict):
            snapshot = gpu_snapshot_from_dict(snapshot)
        return cls(
            node_name=value["node_name"],
            phase=value["phase"],
            snapshot=snapshot,
        )


@dataclass
class GpuScenarioTelemetry(JsonRecord):
    """GPU-specific telemetry associated with one scenario run."""

    fault_type: str
    workers: list[GpuWorkerTelemetry] = field(default_factory=list)
    observations: list[GpuObservation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuScenarioTelemetry:
        return cls(
            fault_type=value["fault_type"],
            workers=[
                item
                if isinstance(item, GpuWorkerTelemetry)
                else GpuWorkerTelemetry.from_dict(item)
                for item in (value.get("workers") or [])
            ],
            observations=[
                item
                if isinstance(item, GpuObservation)
                else GpuObservation.from_dict(item)
                for item in (value.get("observations") or [])
            ],
        )


def gpu_snapshot_from_dict(value: dict[str, Any]) -> GpuSnapshot:
    """Rebuild a typed snapshot, including enum and alternative evidence."""

    measurements = []
    for item in value.get("measurements") or []:
        if isinstance(item, Measurement):
            measurements.append(item)
            continue
        status = item["status"]
        measurements.append(
            Measurement(
                device_uuid=item["device_uuid"],
                scope=item["scope"],
                field=item["field"],
                value=item.get("value"),
                unit=item.get("unit"),
                value_type=item["value_type"],
                source=item["source"],
                observed_at=item["observed_at"],
                status=(
                    status
                    if isinstance(status, Availability)
                    else Availability(status)
                ),
                reason=item.get("reason"),
                alternatives=[
                    evidence
                    if isinstance(evidence, Evidence)
                    else Evidence(
                        source=evidence["source"],
                        value=evidence.get("value"),
                        observed_at=evidence["observed_at"],
                    )
                    for evidence in (item.get("alternatives") or [])
                ],
            )
        )
    return GpuSnapshot(
        context=value.get("context"),
        observed_at=value["observed_at"],
        measurements=measurements,
        errors=list(value.get("errors") or []),
    )
