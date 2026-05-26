import logging
import asyncio

def get_hardware_metrics() -> dict:
    """
    Collects CPU, RAM, Disk and GPU/VRAM metrics.
    Returns a dict injected into WorkflowState.experiment_metrics.
    """
    metrics = {
        "cpu_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "disk_free_gb": None,
        "gpu_available": False,
        "vram_used_gb": None,
        "vram_total_gb": None,
    }

    # --- CPU & RAM ---
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"] = round(vm.used / (1024 ** 3), 2)
        metrics["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
        disk = psutil.disk_usage(".")
        metrics["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)
    except ImportError:
        logging.warning("psutil not installed — CPU/RAM/Disk metrics unavailable.")
    except Exception as e:
        logging.warning(f"Error collecting CPU/RAM metrics: {e}")

    # --- GPU / VRAM ---
    try:
        import torch
        if torch.cuda.is_available():
            metrics["gpu_available"] = True
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            metrics["vram_used_gb"] = round(vram_used, 2)
            metrics["vram_total_gb"] = round(vram_total, 2)
    except ImportError:
        logging.warning("torch not installed — GPU/VRAM metrics unavailable.")
    except Exception as e:
        logging.warning(f"Error collecting GPU metrics: {e}")

    return metrics


def check_hardware_limits(metrics: dict, required_vram_gb: float = 2.0) -> dict:
    """
    Runs pre-flight hardware checks before the planner generates a workflow.
    Returns a dict with warnings/blocks if limits are exceeded.
    """
    warnings = []

    if metrics["gpu_available"] and metrics["vram_total_gb"] is not None:
        vram_free = metrics["vram_total_gb"] - (metrics["vram_used_gb"] or 0)
        if vram_free < required_vram_gb:
            warnings.append(
                f"Low VRAM: {vram_free:.2f} GB free, {required_vram_gb} GB required. "
                "Consider reducing batch size or using CPU training."
            )

    if metrics["ram_total_gb"] is not None:
        ram_free_gb = metrics["ram_total_gb"] - (metrics["ram_used_gb"] or 0)
        if ram_free_gb < 1.0:
            warnings.append(f"Low RAM: Only {ram_free_gb:.2f} GB free.")

    if metrics["disk_free_gb"] is not None and metrics["disk_free_gb"] < 2.0:
        warnings.append(f"Low Disk: Only {metrics['disk_free_gb']:.2f} GB free.")

    return {
        "hardware_ok": len(warnings) == 0,
        "warnings": warnings,
        "metrics": metrics
    }
