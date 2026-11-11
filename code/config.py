"""All assumptions in one place. Anything marked ASSUMED must be justified/cited before submission."""

# ---- Data -------------------------------------------------------------------
HEADLINE_FILE = "2012-2013 Solar home electricity data v2.csv"
DAYS = 365                      # 1 Jul 2012 - 30 Jun 2013
SLOTS = 48                      # half-hours per day
DT_H = 0.5                      # hours per slot

# ---- Battery (ASSUMED, typical modern LFP unit) -----------------------------------
BATT_KWH_PER_HOME_SWEEP = (2.5, 5.0, 10.0)   # equal total capacity in B and C
BATT_C_RATE = 0.5               # max charge/discharge power = C_RATE * capacity (kW)
BATT_ETA_ONE_WAY = 0.95         # round trip ~ 0.90
BATT_SOC_MIN = 0.0              # usable capacity already net of reserve

# ---- PV scaling --------------------------------------------------------------------
# Dataset PV is small (mean 1.68 kWp, 2010-13 installs). Scale 3 ~ 5 kWp, a typical new system.
PV_SCALES = (1.0, 3.0)

# ---- Tariffs (c/kWh, GST inclusive; clock time) ------------------------------------------
# All three energy rates are the Australian Energy Regulator's 2026-27 default-market-offer caps for
# Ausgrid residential customers (AER final determination, 26 May 2026, tariff cap price determination):
#   flat 33.1372 c/kWh; time-of-use peak 60.1966 c/kWh (15:00-21:00, Nov-Mar and Jun-Aug) and
#   off-peak 24.0387 c/kWh (all other times, incl. all day Apr, May, Sep, Oct).
FLAT_C_PER_KWH = 33.1372
TOU_PEAK_MONTHS = (11, 12, 1, 2, 3, 6, 7, 8)
TOU_PEAK_START_H, TOU_PEAK_END_H = 15, 21
TOU_PEAK_C = 60.1966
TOU_OFFPEAK_C = 24.0387
# Earlier versions of this study used retail market-offer rates (54.0 / 21.6 c/kWh); kept for the
# tariff-level sensitivity analysis.
TOU_PEAK_C_MARKET = 54.0
TOU_OFFPEAK_C_MARKET = 21.6

# Feed-in tariff for exports (ASSUMED; swept in sensitivity analysis)
FIT_C_PER_KWH = 5.0

# Demand-charge variant: TOU energy prices plus a charge on the monthly maximum import kW in the
# 15:00-21:00 window. ASSUMED level (A$ per kW per day); swept in sensitivity analysis.
DEMAND_AUD_PER_KW_DAY = 0.40

# Export-limit variant: per-connection export cap, kW per home (ASSUMED, inside the 1.5-5 kW
# range used by Australian network operators). Excess PV is curtailed.
EXPORT_LIMIT_KW_PER_HOME = 1.5

# Demand charge over all 24 hours (variant "tou_demand24"): same A$ per kW per day on the monthly maximum
# import in ANY half-hour, so that charging cannot escape the charge by moving outside 15:00-21:00.
TARIFFS = ("flat", "tou", "tou_demand", "tou_demand24", "tou_exportlimit")

# Heuristic controllers
SPREAD_START_SLOT = 20          # "spread" top-up starts at 10:00 (slot index 20) on peak-price days

# ---- Economics (ASSUMED unless sourced) ---------------------------------------------------
# Home battery installed price after the federal rebate: Solar Choice Battery Price Index, November 2026
# edition (updated 6 Nov 2026): A$798/kWh for a 10 kWh battery with an inverter (A$738/kWh battery only);
# smaller units cost more per kWh. Community battery: ARENA Community Batteries Round 1 average project
# cost A$1.28/Wh = A$1,280/kWh (ARENA media release, 29 June 2024; grant up to A$0.51/Wh).
BATT_COST_HOME_AUD_PER_KWH = 800.0
BATT_COST_COMMUNITY_AUD_PER_KWH = 1280.0
DISCOUNT_RATE = 0.06
BATT_LIFE_MAX_YEARS = 15
BATT_CYCLE_LIFE = 6000          # equivalent full cycles to end of life (LFP, ASSUMED)
