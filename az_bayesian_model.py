import numpy as np

RNG = np.random.default_rng(20260721)

CREDIBILITY_EXPONENT = 2.0
OUTLIER_LAMBDA = 3.0
TAU_FLOOR_STATE = 0.04
TAU_FLOOR_REGION = 0.03
TAU_FLOOR_COUNTY = 0.06
TAU2_CAP = 0.05
PRE_MARICOPA_STATE_DAMPENING = 0.4
ALPHA_DAYOF_AMPLIFICATION = 1.3
LINK_UNCERTAINTY = 0.02
ALPHA_TURNOUT_DAYOF = 1.0
TURNOUT_LINK_UNCERTAINTY = 0.01
TURNOUT_SIGNAL_THRESHOLD = 0.85
TAU_FLOOR_TURNOUT_STATE = 0.03
TAU_FLOOR_TURNOUT_REGION = 0.02
TAU_FLOOR_TURNOUT_COUNTY = 0.05
N_SIMS = 20000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


REGIONS = {
    "Maricopa": ["Maricopa"],
    "Pima": ["Pima"],
    "Rural": ["Pinal", "Yavapai", "Mohave", "Cochise", "Yuma", "Navajo",
              "Coconino", "Gila", "Apache", "Graham", "La Paz", "Santa Cruz", "Greenlee"],
}
COUNTY_REGION = {c: r for r, cs in REGIONS.items() for c in cs}

_CONFIG = {
    "Maricopa":   dict(total=391200, early_share=0.63, early_bso=(0.720, 0.246, 0.034), dayof_bso=(0.753, 0.213, 0.034)),
    "Pima":       dict(total=76500,  early_share=0.65, early_bso=(0.707, 0.234, 0.059), dayof_bso=(0.738, 0.202, 0.059)),
    "Yavapai":    dict(total=46550,  early_share=0.68, early_bso=(0.727, 0.235, 0.038), dayof_bso=(0.760, 0.203, 0.038)),
    "Pinal":      dict(total=38730,  early_share=0.61, early_bso=(0.728, 0.234, 0.039), dayof_bso=(0.761, 0.201, 0.038)),
    "Mohave":     dict(total=30850,  early_share=0.63, early_bso=(0.730, 0.225, 0.045), dayof_bso=(0.762, 0.193, 0.045)),
    "Cochise":    dict(total=13750,  early_share=0.65, early_bso=(0.687, 0.249, 0.064), dayof_bso=(0.718, 0.218, 0.064)),
    "Yuma":       dict(total=11370,  early_share=0.56, early_bso=(0.712, 0.229, 0.059), dayof_bso=(0.744, 0.197, 0.059)),
    "Navajo":     dict(total=11160,  early_share=0.59, early_bso=(0.720, 0.237, 0.043), dayof_bso=(0.752, 0.205, 0.043)),
    "Coconino":   dict(total=9400,   early_share=0.61, early_bso=(0.718, 0.235, 0.047), dayof_bso=(0.751, 0.203, 0.046)),
    "Gila":       dict(total=8500,   early_share=0.70, early_bso=(0.757, 0.205, 0.038), dayof_bso=(0.790, 0.172, 0.038)),
    "Apache":     dict(total=4160,   early_share=0.57, early_bso=(0.713, 0.209, 0.078), dayof_bso=(0.745, 0.178, 0.077)),
    "Graham":     dict(total=3730,   early_share=0.59, early_bso=(0.725, 0.231, 0.044), dayof_bso=(0.757, 0.198, 0.045)),
    "La Paz":     dict(total=1930,   early_share=0.55, early_bso=(0.728, 0.211, 0.061), dayof_bso=(0.760, 0.178, 0.062)),
    "Santa Cruz": dict(total=1590,   early_share=0.56, early_bso=(0.717, 0.218, 0.065), dayof_bso=(0.749, 0.187, 0.065)),
    "Greenlee":   dict(total=570,    early_share=0.57, early_bso=(0.751, 0.185, 0.065), dayof_bso=(0.785, 0.150, 0.064)),
}


def contrast_logodds(x, ref):
    return np.log(np.clip(x, 1e-6, None) / np.clip(ref, 1e-6, None))


class County:
    def __init__(self, name, total, early_share, early_bso, dayof_bso):
        self.name = name
        self.region = COUNTY_REGION[name]
        self.total = total
        self.early_total_proj = total * early_share
        self.dayof_total_proj = total * (1 - early_share)
        self.prior_early_bso = early_bso
        self.prior_dayof_bso = dayof_bso
        self.reported_early = None
        self.reported_dayof = None

    def prior_bso(self, bucket):
        return self.prior_early_bso if bucket == "early" else self.prior_dayof_bso

    def reported(self, bucket):
        return self.reported_early if bucket == "early" else self.reported_dayof

    def report(self, bucket, b, s, o):
        if bucket == "early":
            self.reported_early = (b, s, o)
        else:
            self.reported_dayof = (b, s, o)

    def projected_total(self, bucket):
        return self.early_total_proj if bucket == "early" else self.dayof_total_proj

    def observed_contrast(self, bucket, quantity):
        counts = self.reported(bucket)
        if counts is None:
            return None
        b, s, o = counts
        n = b + s + o
        if n == 0:
            return None
        pb, ps, po = self.prior_bso(bucket)
        if quantity == "B":
            obs = np.log((b + 0.5) / (o + 0.5))
            prior = contrast_logodds(pb, po)
            var = 1 / (b + 0.5) + 1 / (o + 0.5)
        else:
            obs = np.log((s + 0.5) / (o + 0.5))
            prior = contrast_logodds(ps, po)
            var = 1 / (s + 0.5) + 1 / (o + 0.5)
        return obs - prior, var, n

    def observed_turnout(self, bucket):
        counts = self.reported(bucket)
        if counts is None:
            return None
        n = sum(counts)
        proj = self.projected_total(bucket)
        if proj <= 0 or n / proj < TURNOUT_SIGNAL_THRESHOLD:
            return None
        obs = np.log(n / proj)
        var = 1 / (n + 0.5)
        return obs, var, n

    def remaining(self, bucket, adjusted_total=None):
        proj_total = adjusted_total if adjusted_total is not None else self.projected_total(bucket)
        reported = self.reported(bucket)
        reported_n = sum(reported) if reported else 0
        return max(0.0, proj_total - reported_n)


COUNTIES = {name: County(name, **cfg) for name, cfg in _CONFIG.items()}


def dl_pool(estimates, variances):
    estimates = np.array(estimates, dtype=float)
    variances = np.array(variances, dtype=float)
    if len(estimates) == 0:
        return None
    if len(estimates) == 1:
        return estimates[0], variances[0], 0.0

    w_fixed = 1 / variances
    theta_fixed = np.sum(w_fixed * estimates) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (estimates - theta_fixed) ** 2)
    df = len(estimates) - 1
    c = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    tau2 = min(tau2, TAU2_CAP)

    w_random = 1 / (variances + tau2)
    theta_random = np.sum(w_random * estimates) / np.sum(w_random)
    var_random = 1 / np.sum(w_random)
    return theta_random, var_random, tau2


def outlier_downweight(estimates, variances, pooled_mean, pooled_sd):
    kept_var = []
    for e, v in zip(estimates, variances):
        z = abs(e - pooled_mean) / max(pooled_sd, 1e-6)
        if z > OUTLIER_LAMBDA:
            kept_var.append(v * (z / OUTLIER_LAMBDA) ** 2)
        else:
            kept_var.append(v)
    return list(estimates), kept_var


def coverage(bucket, county_names=None):
    names = county_names if county_names is not None else list(COUNTIES.keys())
    total_proj = sum(COUNTIES[n].projected_total(bucket) for n in names)
    total_reported = 0
    counties_reporting = 0
    for n in names:
        rep = COUNTIES[n].reported(bucket)
        if rep is not None:
            total_reported += sum(rep)
            counties_reporting += 1
    vote_pct = total_reported / total_proj if total_proj else 0
    county_pct = counties_reporting / len(names) if names else 0
    return min(vote_pct, county_pct) ** CREDIBILITY_EXPONENT


def hierarchical_pool(obs, bucket, floor_state, floor_region, floor_county):
    if not obs:
        return None

    ests = [v[0] for v in obs.values()]
    varss = [v[1] for v in obs.values()]

    pooled = dl_pool(ests, varss)
    state_mean, state_var, _ = pooled
    ests2, varss2 = outlier_downweight(ests, varss, state_mean, np.sqrt(max(state_var, 1e-9)))
    state_mean, state_var, _ = dl_pool(ests2, varss2)
    state_var = state_var if len(ests2) >= 2 else floor_state ** 2
    state_var = max(state_var, floor_state ** 2)

    shrink_state = coverage(bucket)
    state_mean *= shrink_state

    if COUNTIES["Maricopa"].reported(bucket) is None:
        state_mean *= PRE_MARICOPA_STATE_DAMPENING

    region_means, region_vars = {}, {}
    for region, names in REGIONS.items():
        residuals, rvars = [], []
        for name, (val, var, n) in obs.items():
            if COUNTY_REGION[name] == region:
                residuals.append(val - state_mean)
                rvars.append(var)
        if residuals:
            rp = dl_pool(residuals, rvars)
            shrink_region = coverage(bucket, county_names=names)
            region_means[region] = rp[0] * shrink_region
            region_vars[region] = rp[1] if len(residuals) >= 2 else floor_region ** 2
            region_vars[region] = max(region_vars[region], floor_region ** 2)
        else:
            region_means[region] = 0.0
            region_vars[region] = floor_region ** 2 * 2

    resids = []
    for name, (val, var, n) in obs.items():
        resids.append(val - state_mean - region_means[COUNTY_REGION[name]])
    county_tau2 = max(np.var(resids) if len(resids) > 1 else 0.0, floor_county ** 2)

    return dict(state_mean=state_mean, state_var=state_var,
                region_means=region_means, region_vars=region_vars,
                county_tau2=county_tau2, n_reporting=len(obs))


def empty_factor(floor_state, floor_region, floor_county, wide=False):
    mult = 4 if wide else 1
    return dict(state_mean=0.0, state_var=(floor_state ** 2) * mult,
                region_means={r: 0.0 for r in REGIONS},
                region_vars={r: (floor_region ** 2) * mult for r in REGIONS},
                county_tau2=floor_county ** 2, n_reporting=0)


def blend(m1, v1, m2, v2):
    w1, w2 = 1 / v1, 1 / v2
    return (m1 * w1 + m2 * w2) / (w1 + w2), 1 / (w1 + w2)


def get_early_factor(quantity):
    obs = {}
    for name, c in COUNTIES.items():
        r = c.observed_contrast("early", quantity) if quantity in ("B", "S") else c.observed_turnout("early")
        if r is not None:
            obs[name] = r
    if quantity == "T":
        floors = (TAU_FLOOR_TURNOUT_STATE, TAU_FLOOR_TURNOUT_REGION, TAU_FLOOR_TURNOUT_COUNTY)
    else:
        floors = (TAU_FLOOR_STATE, TAU_FLOOR_REGION, TAU_FLOOR_COUNTY)
    pooled = hierarchical_pool(obs, "early", *floors)
    if pooled is None:
        return empty_factor(*floors)
    return pooled


def get_dayof_factor(quantity):
    early_factors = get_early_factor(quantity)

    obs = {}
    for name, c in COUNTIES.items():
        r = c.observed_contrast("dayof", quantity) if quantity in ("B", "S") else c.observed_turnout("dayof")
        if r is not None:
            obs[name] = r
    if quantity == "T":
        floors = (TAU_FLOOR_TURNOUT_STATE, TAU_FLOOR_TURNOUT_REGION, TAU_FLOOR_TURNOUT_COUNTY)
        alpha, link_unc = ALPHA_TURNOUT_DAYOF, TURNOUT_LINK_UNCERTAINTY
    else:
        floors = (TAU_FLOOR_STATE, TAU_FLOOR_REGION, TAU_FLOOR_COUNTY)
        alpha, link_unc = ALPHA_DAYOF_AMPLIFICATION, LINK_UNCERTAINTY

    dayof_direct = hierarchical_pool(obs, "dayof", *floors)

    inf_state_mean = alpha * early_factors["state_mean"]
    inf_state_var = (alpha ** 2) * early_factors["state_var"] + link_unc
    inf_region_means = {g: alpha * m for g, m in early_factors["region_means"].items()}
    inf_region_vars = {g: (alpha ** 2) * v + link_unc for g, v in early_factors["region_vars"].items()}

    if dayof_direct is None:
        return dict(state_mean=inf_state_mean, state_var=max(inf_state_var, floors[0] ** 2),
                    region_means=inf_region_means,
                    region_vars={g: max(v, floors[1] ** 2) for g, v in inf_region_vars.items()},
                    county_tau2=floors[2] ** 2, n_reporting=0)

    state_mean, state_var = blend(inf_state_mean, inf_state_var,
                                   dayof_direct["state_mean"], dayof_direct["state_var"])
    region_means, region_vars = {}, {}
    for g in REGIONS:
        region_means[g], region_vars[g] = blend(
            inf_region_means[g], inf_region_vars[g],
            dayof_direct["region_means"][g], dayof_direct["region_vars"][g])

    county_tau2 = max(dayof_direct["county_tau2"], floors[2] ** 2)

    return dict(state_mean=state_mean, state_var=max(state_var, floors[0] ** 2),
                region_means=region_means,
                region_vars={g: max(v, floors[1] ** 2) for g, v in region_vars.items()},
                county_tau2=county_tau2, n_reporting=dayof_direct["n_reporting"])


RHO_BS = -0.85


def correlated_pair(mean1, sd1, mean2, sd2, rho, n_sims):
    z1 = RNG.normal(0, 1, n_sims)
    z2 = RNG.normal(0, 1, n_sims)
    x1 = mean1 + sd1 * z1
    x2 = mean2 + sd2 * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
    return x1, x2


def draw_factor(factor, n_sims):
    state_draw = RNG.normal(factor["state_mean"], np.sqrt(factor["state_var"]), n_sims)
    region_draws = {
        g: RNG.normal(factor["region_means"][g], np.sqrt(factor["region_vars"][g]), n_sims)
        for g in REGIONS
    }
    county_tau = np.sqrt(factor["county_tau2"])
    return state_draw, region_draws, county_tau


def draw_factor_pair_BS(factor_B, factor_S, rho, n_sims):
    state_B, state_S = correlated_pair(
        factor_B["state_mean"], np.sqrt(factor_B["state_var"]),
        factor_S["state_mean"], np.sqrt(factor_S["state_var"]),
        rho, n_sims
    )
    region_B, region_S = {}, {}
    for g in REGIONS:
        rb, rs = correlated_pair(
            factor_B["region_means"][g], np.sqrt(factor_B["region_vars"][g]),
            factor_S["region_means"][g], np.sqrt(factor_S["region_vars"][g]),
            rho, n_sims
        )
        region_B[g] = rb
        region_S[g] = rs
    tau_B = np.sqrt(factor_B["county_tau2"])
    tau_S = np.sqrt(factor_S["county_tau2"])
    return state_B, state_S, region_B, region_S, tau_B, tau_S


def simulate(n_sims=N_SIMS):
    factors = {}
    for bucket in ("early", "dayof"):
        for quantity in ("B", "S", "T"):
            factors[(bucket, quantity)] = (
                get_early_factor(quantity) if bucket == "early" else get_dayof_factor(quantity)
            )

    draws = {}
    for bucket in ("early", "dayof"):
        draws[(bucket, "T")] = draw_factor(factors[(bucket, "T")], n_sims)
        state_B, state_S, region_B, region_S, tau_B, tau_S = draw_factor_pair_BS(
            factors[(bucket, "B")], factors[(bucket, "S")], RHO_BS, n_sims
        )
        draws[(bucket, "B")] = (state_B, region_B, tau_B)
        draws[(bucket, "S")] = (state_S, region_S, tau_S)

    final_B = np.zeros(n_sims)
    final_S = np.zeros(n_sims)
    final_O = np.zeros(n_sims)

    for c in COUNTIES.values():
        for bucket in ("early", "dayof"):
            reported = c.reported(bucket)
            reported_n = sum(reported) if reported else 0
            if reported:
                final_B += reported[0]
                final_S += reported[1]
                final_O += reported[2]

            state_T, region_T, tau_T = draws[(bucket, "T")]
            county_noise_T = RNG.normal(0, tau_T, n_sims)
            turnout_shift = state_T + region_T[c.region] + county_noise_T
            adjusted_total = c.projected_total(bucket) * np.exp(turnout_shift)
            remaining = np.maximum(0.0, adjusted_total - reported_n)

            pb, ps, po = c.prior_bso(bucket)
            prior_B_logodds = contrast_logodds(pb, po)
            prior_S_logodds = contrast_logodds(ps, po)

            state_B, region_B, tau_B = draws[(bucket, "B")]
            state_S, region_S, tau_S = draws[(bucket, "S")]
            county_noise_B, county_noise_S = correlated_pair(0, tau_B, 0, tau_S, RHO_BS, n_sims)

            shift_B = state_B + region_B[c.region] + county_noise_B
            shift_S = state_S + region_S[c.region] + county_noise_S

            sampling_sd = 1 / np.sqrt(np.maximum(remaining, 1))
            sampling_noise_B, sampling_noise_S = correlated_pair(
                0, sampling_sd, 0, sampling_sd, RHO_BS, n_sims
            )

            logodds_B = prior_B_logodds + shift_B + sampling_noise_B
            logodds_S = prior_S_logodds + shift_S + sampling_noise_S

            uB = np.exp(logodds_B)
            uS = np.exp(logodds_S)
            uO = 1.0
            total_u = uB + uS + uO

            B_share = uB / total_u
            S_share = uS / total_u
            O_share = uO / total_u

            final_B += remaining * B_share
            final_S += remaining * S_share
            final_O += remaining * O_share

    total = final_B + final_S + final_O
    return final_B, final_S, final_O, total


def report_status():
    B, S, O, total = simulate()
    B_pct, S_pct, O_pct = B / total * 100, S / total * 100, O / total * 100
    margin = B - S
    win_prob_B = np.mean(B > S) * 100

    print("=" * 55)
    print("AZ REPUBLICAN PRIMARY - LIVE PROJECTION")
    print("=" * 55)
    print(f"P(B wins):        {win_prob_B:.2f}%")
    print(f"B share:          median {np.median(B_pct):.2f}%  "
          f"[{np.percentile(B_pct, 2.5):.2f}, {np.percentile(B_pct, 97.5):.2f}]")
    print(f"S share:          median {np.median(S_pct):.2f}%  "
          f"[{np.percentile(S_pct, 2.5):.2f}, {np.percentile(S_pct, 97.5):.2f}]")
    print(f"O share:          median {np.median(O_pct):.2f}%")
    print(f"Total votes:      median {np.median(total):,.0f}  "
          f"[{np.percentile(total, 2.5):,.0f}, {np.percentile(total, 97.5):,.0f}]")
    print(f"B-S margin:       median {np.median(margin):,.0f} votes  "
          f"[{np.percentile(margin, 2.5):,.0f}, {np.percentile(margin, 97.5):,.0f}]")


def county_point_estimate_remaining(name):
    county = COUNTIES[name]
    early_f = {q: get_early_factor(q) for q in ("B", "S", "T")}
    dayof_f = {q: get_dayof_factor(q) for q in ("B", "S", "T")}

    totals = {"B": 0.0, "S": 0.0, "O": 0.0}
    for bucket, factors in (("early", early_f), ("dayof", dayof_f)):
        reported = county.reported(bucket)
        reported_n = sum(reported) if reported else 0

        shift_T = factors["T"]["state_mean"] + factors["T"]["region_means"][county.region]
        adjusted_total = county.projected_total(bucket) * np.exp(shift_T)
        remaining = max(0.0, adjusted_total - reported_n)
        if remaining <= 0:
            continue

        pb, ps, po = county.prior_bso(bucket)
        prior_B_logodds = contrast_logodds(pb, po)
        prior_S_logodds = contrast_logodds(ps, po)
        shift_B = factors["B"]["state_mean"] + factors["B"]["region_means"][county.region]
        shift_S = factors["S"]["state_mean"] + factors["S"]["region_means"][county.region]

        uB = np.exp(prior_B_logodds + shift_B)
        uS = np.exp(prior_S_logodds + shift_S)
        uO = 1.0
        total_u = uB + uS + uO

        totals["B"] += remaining * uB / total_u
        totals["S"] += remaining * uS / total_u
        totals["O"] += remaining * uO / total_u

    return totals


def decomposition():
    real_B = 0.0
    baseline_remaining_B = 0.0
    adjusted_remaining_B = 0.0

    for name, county in COUNTIES.items():
        for bucket in ("early", "dayof"):
            reported = county.reported(bucket)
            reported_n = sum(reported) if reported else 0
            if reported:
                real_B += reported[0]

            pure_prior_total = county.projected_total(bucket)
            pure_remaining_n = max(0.0, pure_prior_total - reported_n)
            pb, ps, po = county.prior_bso(bucket)
            baseline_remaining_B += pure_remaining_n * pb

        adjusted_remaining_B += county_point_estimate_remaining(name)["B"]

    total_projected_B = real_B + adjusted_remaining_B
    modeled_adjustment_B = adjusted_remaining_B - baseline_remaining_B

    return {
        "realVotesB": real_B,
        "baselineRemainingB": baseline_remaining_B,
        "modeledAdjustmentB": modeled_adjustment_B,
        "totalProjectedB": total_projected_B,
        "realVotesPct": real_B / total_projected_B if total_projected_B else 0,
        "baselinePct": baseline_remaining_B / total_projected_B if total_projected_B else 0,
        "modeledAdjustmentPct": modeled_adjustment_B / total_projected_B if total_projected_B else 0,
    }


def snapshot(n_sims=N_SIMS):
    counties_out = {}
    for name, county in COUNTIES.items():
        early = county.reported("early") or (0, 0, 0)
        dayof = county.reported("dayof") or (0, 0, 0)
        reported = {
            "B": early[0] + dayof[0],
            "S": early[1] + dayof[1],
            "O": early[2] + dayof[2],
        }
        reported["total"] = reported["B"] + reported["S"] + reported["O"]

        remaining = county_point_estimate_remaining(name)
        remaining["total"] = remaining["B"] + remaining["S"] + remaining["O"]

        counties_out[name] = {"reported": reported, "remaining": remaining}

    B, S, O, total = simulate(n_sims)
    bShare, sShare, oShare = B / total * 100, S / total * 100, O / total * 100
    win_prob_B = float(np.mean(B > S) * 100)

    reported_total = sum(c["reported"]["total"] for c in counties_out.values())
    projected_total = sum(COUNTIES[n].total for n in COUNTIES)

    statewide_out = {
        "pBiggs": win_prob_B,
        "bShareMedian": float(np.median(bShare)),
        "bShareP25": float(np.percentile(bShare, 25)),
        "bShareP75": float(np.percentile(bShare, 75)),
        "sShareMedian": float(np.median(sShare)),
        "sShareP25": float(np.percentile(sShare, 25)),
        "sShareP75": float(np.percentile(sShare, 75)),
        "oShareMedian": float(np.median(oShare)),
        "totalMedian": float(np.median(total)),
        "reportedTotal": reported_total,
        "projectedTotal": projected_total,
        "pctIn": reported_total / projected_total if projected_total else 0,
    }

    import time
    return {
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counties": counties_out,
        "statewide": statewide_out,
        "decomposition": decomposition(),
    }


if __name__ == "__main__":
    report_status()
