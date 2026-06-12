"""Modtran6NC class — parses MODTRAN 6 .nc output into a describer_df-compatible DataFrame."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import simpson
import xarray as xr

try:
    from cloudpathlib import AnyPath
except ImportError:
    AnyPath = None


def _as_path(p):
    """Normalize a str/Path/S3Path to a path object with .open() and .name."""
    if isinstance(p, str):
        return AnyPath(p) if AnyPath is not None else Path(p)
    return p


_DEFAULT_SRF_DIR = Path(__file__).parent.parent / "data" / "SRF"

# Maps CERES_TRMM_Scene_ID (int) → one of the 5 unfiltering scene type strings.
# Confirmed: 5=Clear Ocean, 18=Cloudy Ocean. Extend when new Scene_IDs appear in the full dataset.
CERES_SCENE_MAP: dict[int, str] = {
    5: "Clear Ocean",
    18: "Cloudy Ocean",
}

DCC_CLDC_THRESHOLD = 0.10
DCC_CLD_OT_THRESHOLD = 10.0


class Modtran6NC:
    def __init__(self, nc_path, srf_path=None):
        self.nc_path = _as_path(nc_path)
        self.srf_path = _as_path(srf_path) if srf_path is not None else _DEFAULT_SRF_DIR
        ds = xr.open_dataset(nc_path)
        try:
            self.describer_df = self._build_describer_df(ds)
        finally:
            ds.close()

    def load_srf(self, channel: str, version: str = "0-0-1"):
        # TODO: extract to shared srf_utils module alongside tp7.py
        srf_file = self.srf_path / f"libera_srf_{channel}_v{version}.csv"
        with srf_file.open("r") as f:
            return pd.read_csv(f, header=1, names=["Wavelength", "Response"])

    def get_interpolated_srf(self, channel: str, interpolation_points=np.linspace(0.3, 100, 500), version: str = "0-0-1"):
        # TODO: extract to shared srf_utils module alongside tp7.py
        base_srf_data = self.load_srf(channel, version)
        return np.interp(interpolation_points, base_srf_data.Wavelength, base_srf_data.Response)

    def _determine_scene_and_cloud(self, ds: xr.Dataset):
        cldc = float(ds["FV3_CLDC"].values)
        cld_ot = float(ds["FV3_CLD_OT"].values)

        # DCC takes precedence — check thresholds before consulting Scene_ID
        if cldc >= DCC_CLDC_THRESHOLD and cld_ot > DCC_CLD_OT_THRESHOLD:
            return "Deep Convective Cloud", 1

        scene_id = int(ds["CERES_TRMM_Scene_ID"].values)
        scene_type = CERES_SCENE_MAP.get(scene_id)
        if scene_type is None:
            raise ValueError(
                f"Unknown CERES_TRMM_Scene_ID: {scene_id}. Add it to CERES_SCENE_MAP in tp7/modtran6.py."
            )
        cloud = 1 if cldc > 0 else 0
        return scene_type, cloud

    def _build_describer_df(self, ds: xr.Dataset) -> pd.DataFrame:
        scene_type, cloud = self._determine_scene_and_cloud(ds)

        sza_vals = ds["CERES_TRMM_SZA"].values   # (9,)
        vza_vals = ds["CERES_TRMM_VZA"].values   # (9,)
        raa_vals = ds["CERES_TRMM_RAA"].values   # (10,)

        # Build rows in same order as the (vza, raa, sza, wavelength) array layout
        rows = []
        for vza in vza_vals:
            for raa in raa_vals:
                for sza in sza_vals:
                    rows.append({"SZA": float(sza), "VZA": float(vza), "RAZ": float(raa)})

        df = pd.DataFrame(rows)
        df["Scene"] = scene_type
        df["Cloud"] = cloud
        df["Run #"] = range(len(df))

        self._integrate_radiances(ds, df)
        return df

    def _integrate_radiances(self, ds: xr.Dataset, df: pd.DataFrame) -> None:
        sw_spec = ds["MODTRAN6_SPECTRAL_RADIANCE_TOA_SW_WVL_CERES_TRMM"].values  # (vza, raa, sza, wvl_sw)
        lw_spec = ds["MODTRAN6_SPECTRAL_RADIANCE_TOA_LW_WVL_CERES_TRMM"].values  # (vza, raa, sza, wvl_lw)
        # Spectral radiance is in W m⁻² sr⁻¹ nm⁻¹; integrate over nm to get W m⁻² sr⁻¹.
        # SRF files use µm, so interpolate SRFs using µm wavelengths only.
        wvl_sw_nm = ds.coords["wavelength_sw"].values          # nm — used as integration axis
        wvl_lw_nm = ds.coords["wavelength_lw"].values
        wvl_sw_um = wvl_sw_nm / 1000.0                        # µm — used for SRF lookup only
        wvl_lw_um = wvl_lw_nm / 1000.0

        sw_srf       = self.get_interpolated_srf("sw",    wvl_sw_um)
        ssw_srf      = self.get_interpolated_srf("ssw",   wvl_sw_um)
        lw_srf       = self.get_interpolated_srf("lw",    wvl_lw_um)
        total_srf_sw = self.get_interpolated_srf("total", wvl_sw_um)
        total_srf_lw = self.get_interpolated_srf("total", wvl_lw_um)
        ssw_passband = (ssw_srf > 0).astype(float)

        # Vectorized integration over all angle combos; result shape (vza, raa, sza)
        sw_unfilt  = simpson(sw_spec,                   x=wvl_sw_nm, axis=-1)
        lw_unfilt  = simpson(lw_spec,                   x=wvl_lw_nm, axis=-1)
        ssw_unfilt = simpson(sw_spec * ssw_passband,    x=wvl_sw_nm, axis=-1)
        sw_filt    = simpson(sw_spec * sw_srf,          x=wvl_sw_nm, axis=-1)
        lw_filt    = simpson(lw_spec * lw_srf,          x=wvl_lw_nm, axis=-1)
        ssw_filt   = simpson(sw_spec * ssw_srf,         x=wvl_sw_nm, axis=-1)
        tot_unfilt = sw_unfilt + lw_unfilt
        tot_filt   = (simpson(sw_spec * total_srf_sw, x=wvl_sw_nm, axis=-1) +
                      simpson(lw_spec * total_srf_lw, x=wvl_lw_nm, axis=-1))

        # Flatten to 1D in C order (vza slowest) to match df row order
        df["Shortwave Unfiltered Rads (Integrated)"]       = sw_unfilt.reshape(-1)
        df["Longwave Unfiltered Rads (Integrated)"]        = lw_unfilt.reshape(-1)
        df["Total Unfiltered Rads (Integrated)"]           = tot_unfilt.reshape(-1)
        df["Split Shortwave Unfiltered Rads (Integrated)"] = ssw_unfilt.reshape(-1)
        df["Shortwave Filtered Rads (Integrated)"]         = sw_filt.reshape(-1)
        df["Longwave Filtered Rads (Integrated)"]          = lw_filt.reshape(-1)
        df["Split Shortwave Filtered Rads (Integrated)"]   = ssw_filt.reshape(-1)
        df["Total Filtered Rads (Integrated)"]             = tot_filt.reshape(-1)