from __future__ import annotations

import math

from .models import RadioConfig

C_LIGHT = 299_792_458.0


def eirp_dbm(radio: RadioConfig) -> float:
    return radio.tx_power_dbm + radio.tx_antenna_gain_dbi


def free_space_path_loss_db(distance_m: float, carrier_hz: float) -> float:
    distance_m = max(distance_m, 1e-3)
    return 20 * math.log10(4 * math.pi * distance_m * carrier_hz / C_LIGHT)


def noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    return -174.0 + 10 * math.log10(bandwidth_hz) + noise_figure_db


def received_power_dbm(distance_m: float, radio: RadioConfig) -> float:
    return eirp_dbm(radio) + radio.rx_antenna_gain_dbi - free_space_path_loss_db(distance_m, radio.carrier_hz)


def snr_at_distance_db(distance_m: float, radio: RadioConfig) -> float:
    return received_power_dbm(distance_m, radio) - noise_floor_dbm(radio.bandwidth_hz, radio.noise_figure_db)

