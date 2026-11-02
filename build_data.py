#!/usr/bin/env python3
"""NFL Stats Tracker data pipeline. Downloads free nflverse data and writes data.json.
Usage: python3 build_data.py [season] [outdir]"""
import sys, os, json, datetime, urllib.request
import pandas as pd, numpy as np

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(OUT, "raw"); os.makedirs(RAW, exist_ok=True)
B = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {
    "pw": f"{B}/stats_player/stats_player_week_{SEASON}.parquet",
    "pbp": f"{B}/pbp/play_by_play_{SEASON}.parquet",
    "ros": f"{B}/rosters/roster_{SEASON}.parquet",
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
}
paths = {}
for k, u in FILES.items():
    p = os.path.join(RAW, os.path.basename(u)); paths[k] = p
    urllib.request.urlretrieve(u, p)

pw = pd.read_parquet(paths["pw"]); pw = pw[pw.season_type == "REG"]
pbp = pd.read_parquet(paths["pbp"]); pbp = pbp[pbp.season_type == "REG"]
ros = pd.read_parquet(paths["ros"])
games = pd.read_csv(paths["games"]); games = games[(games.season == SEASON) & (games.game_type == "REG")]

def r(x, n=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        return round(float(x), n)
    except Exception: return None

last_week = int(pw.week.max())
unplayed = games[games.result.isna()]
# upcoming week = first week where most games are still unplayed
next_week = None
for wk_, gg in games.groupby("week"):
    if gg.result.isna().sum() > len(gg) / 2: next_week = int(wk_); break

# ---------------- players ----------------
F = ["completions","attempts","passing_yards","passing_tds","passing_interceptions","sacks_suffered",
     "carries","rushing_yards","rushing_tds","receptions","targets","receiving_yards","receiving_tds",
     "receiving_yards_after_catch","def_tackles_solo","def_tackle_assists","def_tackles_for_loss","def_sacks",
     "def_qb_hits","def_interceptions","def_pass_defended","def_fumbles_forced","def_tds"]
KEYS = ["cmp","att","pyd","ptd","int","sk","car","ryd","rtd","rec","tgt","recyd","rectd","yac",
        "tkl","ast","tfl","sacks","qbh","dint","pd","ff","dtd"]
pw[F] = pw[F].fillna(0)
pw = pw[(pw[F].abs().sum(axis=1) > 0)]
latest = pw.sort_values("week").groupby("player_id").tail(1).set_index("player_id")
rosinfo = ros.sort_values("week").groupby("gsis_id").tail(1).set_index("gsis_id")

# play-by-play derived strengths
pas = pbp[(pbp.pass_attempt == 1) & (pbp.sack != 1) & pbp.air_yards.notna()].copy()
pas["depth"] = np.where(pas.air_yards >= 20, "deep", np.where(pas.air_yards >= 10, "mid", "short"))
run = pbp[(pbp.rush_attempt == 1) & (pbp.qb_scramble != 1)].copy()

def share(s):
    c = s.value_counts(normalize=True); return {k: r(v * 100, 0) for k, v in c.items()}

recv = {}
for pid, g in pas[pas.receiver_player_id.notna()].groupby("receiver_player_id"):
    if len(g) < 5: continue
    comp = g[g.complete_pass == 1]
    deep = g[g.depth == "deep"]
    recv[pid] = {
        "n": len(g), "adot": r(g.air_yards.mean()),
        "depth": share(g.depth), "side": share(g.pass_location.dropna()),
        "catch": r(g.complete_pass.mean() * 100, 0),
        "yac": r(comp.yards_after_catch.mean()) if len(comp) else None,
        "deepCatch": r(deep.complete_pass.mean() * 100, 0) if len(deep) >= 3 else None,
        "epa": r(g.epa.mean(), 2),
    }
rush = {}
for pid, g in run[run.rusher_player_id.notna()].groupby("rusher_player_id"):
    if len(g) < 10: continue
    gap = g.run_gap.fillna("middle"); loc = g.run_location.fillna("middle")
    zone = np.where(gap == "end", "outside", np.where(gap == "tackle", "tackle", "inside"))
    rush[pid] = {"n": len(g), "side": share(loc), "zone": share(pd.Series(zone)),
                 "ypc": r(g.yards_gained.mean()), "explosive": r((g.yards_gained >= 10).mean() * 100, 0),
                 "stuff": r((g.yards_gained <= 0).mean() * 100, 0), "epa": r(g.epa.mean(), 2)}
qbs = {}
for pid, g in pas[pas.passer_player_id.notna()].groupby("passer_player_id"):
    if len(g) < 30: continue
    dz = {}
    for d, gg in g.groupby("depth"):
        dz[d] = {"pct": r(len(gg) / len(g) * 100, 0), "cmp": r(gg.complete_pass.mean() * 100, 0), "epa": r(gg.epa.mean(), 2)}
    qbs[pid] = {"n": len(g), "adot": r(g.air_yards.mean()), "depth": dz, "side": share(g.pass_location.dropna()),
                "epa": r(g.epa.mean(), 2)}

players = []
for pid, g in pw.groupby("player_id"):
    L = latest.loc[pid]
    tot = g[F].sum()
    weeks = [[int(w.week), w.opponent_team] + [r(w[f], 1) for f in F] for _, w in g.sort_values("week").iterrows()]
    ri = rosinfo.loc[pid] if pid in rosinfo.index else None
    p = {"id": pid, "name": L.player_display_name, "pos": L.position, "grp": L.position_group, "team": L.team,
         "gp": int(g.week.nunique()), "tot": {k: r(tot[f], 1) for k, f in zip(KEYS, F)}, "wk": weeks}
    if ri is not None:
        p["num"] = None if pd.isna(ri.jersey_number) else int(ri.jersey_number)
        p["ht"] = None if pd.isna(ri.height) else int(ri.height)
        p["wt"] = None if pd.isna(ri.weight) else int(ri.weight)
        p["col"] = None if pd.isna(ri.college) else ri.college
        p["exp"] = None if pd.isna(ri.years_exp) else int(ri.years_exp)
    epa = g[["passing_epa","rushing_epa","receiving_epa"]].fillna(0).sum()
    p["epa"] = {"pass": r(epa.passing_epa), "rush": r(epa.rushing_epa), "rec": r(epa.receiving_epa)}
    if pid in recv: p["recv"] = recv[pid]
    if pid in rush: p["rush"] = rush[pid]
    if pid in qbs: p["qb"] = qbs[pid]
    players.append(p)

# ---------------- teams ----------------
plays = pbp[(pbp.play_type.isin(["pass", "run"])) & pbp.epa.notna()]
TEAMS = sorted(plays.posteam.dropna().unique())
# receiver position for defense-vs-position
pos_map = rosinfo["position"].to_dict()
pas["rpos"] = pas.receiver_player_id.map(pos_map)

def lg_rank(d, key, asc):
    vals = sorted([(v[key], t) for t, v in d.items() if v[key] is not None], reverse=not asc)
    for i, (_, t) in enumerate(vals): d[t][key + "_rk"] = i + 1

done = games[games.result.notna()]
teams = {}
for t in TEAMS:
    o = plays[plays.posteam == t]; d = plays[plays.defteam == t]
    op, dp = o[o.play_type == "pass"], d[d.play_type == "pass"]
    orn, drn = o[o.play_type == "run"], d[d.play_type == "run"]
    hg = done[done.home_team == t]; ag = done[done.away_team == t]
    pf = hg.home_score.sum() + ag.away_score.sum(); pa = hg.away_score.sum() + ag.home_score.sum()
    w = (hg.result > 0).sum() + (ag.result < 0).sum(); l = (hg.result < 0).sum() + (ag.result > 0).sum()
    tie = (hg.result == 0).sum() + (ag.result == 0).sum(); gp = len(hg) + len(ag)
    tp = pas[pas.defteam == t]
    dz = {}
    for z in ["short", "mid", "deep"]:
        gg = tp[tp.depth == z]
        dz[z] = {"n": len(gg), "epa": r(gg.epa.mean(), 3), "ypa": r(gg.yards_gained.mean()), "cmp": r(gg.complete_pass.mean() * 100, 0)}
    ds = {}
    for z in ["left", "middle", "right"]:
        gg = tp[tp.pass_location == z]
        ds[z] = {"n": len(gg), "epa": r(gg.epa.mean(), 3), "ypa": r(gg.yards_gained.mean())}
    dpos = {}
    for z in ["WR", "TE", "RB"]:
        gg = tp[tp.rpos == z]
        dpos[z] = {"n": len(gg), "epa": r(gg.epa.mean(), 3), "ypt": r(gg.yards_gained.mean())}
    rr = drn.copy(); gap = rr.run_gap.fillna("middle")
    rr["zone"] = np.where(gap == "end", "outside", np.where(gap == "tackle", "tackle", "inside"))
    drz = {z: {"n": int((rr.zone == z).sum()), "epa": r(rr[rr.zone == z].epa.mean(), 3), "ypc": r(rr[rr.zone == z].yards_gained.mean())}
           for z in ["inside", "tackle", "outside"]}
    teams[t] = {
        "team": t, "w": int(w), "l": int(l), "t": int(tie), "gp": int(gp), "pf": int(pf), "pa": int(pa),
        "off": r(o.epa.mean(), 3), "def": r(d.epa.mean(), 3),
        "passOff": r(op.epa.mean(), 3), "rushOff": r(orn.epa.mean(), 3),
        "passDef": r(dp.epa.mean(), 3), "rushDef": r(drn.epa.mean(), 3),
        "offYpp": r(o.yards_gained.mean()), "defYpp": r(d.yards_gained.mean()),
        "passYpg": r(op.yards_gained.sum() / max(gp, 1)), "rushYpg": r(orn.yards_gained.sum() / max(gp, 1)),
        "passYpgA": r(dp.yards_gained.sum() / max(gp, 1)), "rushYpgA": r(drn.yards_gained.sum() / max(gp, 1)),
        "offSucc": r(o.success.mean() * 100, 1), "defSucc": r(d.success.mean() * 100, 1),
        "sacks": int(d.sack.fillna(0).sum()), "takeaways": int(d.interception.fillna(0).sum() + d.fumble_lost.fillna(0).sum()),
        "giveaways": int(o.interception.fillna(0).sum() + o.fumble_lost.fillna(0).sum()),
        "dDepth": dz, "dSide": ds, "dPos": dpos, "dRun": drz,
    }
for t, v in teams.items():
    v["net"] = r((v["off"] or 0) - (v["def"] or 0), 3)
    v["pdpg"] = r((v["pf"] - v["pa"]) / max(v["gp"], 1))
    wp = (v["w"] + 0.5 * v["t"]) / max(v["gp"], 1)
    # power score: efficiency (60%), point diff (25%), record (15%), on z-scales
    v["_wp"] = wp
nets = np.array([v["net"] for v in teams.values()]); pds = np.array([v["pdpg"] for v in teams.values()]); wps = np.array([v["_wp"] for v in teams.values()])
z = lambda a, x: (x - a.mean()) / (a.std() or 1)
for v in teams.values():
    v["power"] = r(50 + 10 * (0.6 * z(nets, v["net"]) + 0.25 * z(pds, v["pdpg"]) + 0.15 * z(wps, v["_wp"])), 1)
    del v["_wp"]
for k, asc in [("power", False), ("off", False), ("def", True), ("passOff", False), ("rushOff", False),
               ("passDef", True), ("rushDef", True), ("net", False)]:
    lg_rank(teams, k, asc)

# league baselines for matchup math
lg = {"depth": {zn: pas[pas.depth == zn].epa.mean() for zn in ["short", "mid", "deep"]},
      "side": {zn: pas[pas.pass_location == zn].epa.mean() for zn in ["left", "middle", "right"]},
      "pos": {zn: pas[pas.rpos == zn].epa.mean() for zn in ["WR", "TE", "RB"]}}
rr = run.copy(); gap = rr.run_gap.fillna("middle")
rr["zone"] = np.where(gap == "end", "outside", np.where(gap == "tackle", "tackle", "inside"))
lg["run"] = {zn: rr[rr.zone == zn].epa.mean() for zn in ["inside", "tackle", "outside"]}
# rank each defense within zones (1 = best defense)
for cat, key in [("depth", "dDepth"), ("side", "dSide"), ("pos", "dPos"), ("run", "dRun")]:
    zones = list(lg[cat].keys())
    for zn in zones:
        vals = sorted([(teams[t][key][zn]["epa"], t) for t in teams if teams[t][key][zn]["epa"] is not None])
        for i, (_, t) in enumerate(vals): teams[t][key][zn]["rk"] = i + 1

# ---------------- matchups for next week ----------------
LBL = {"short": "short", "mid": "intermediate", "deep": "deep",
       "left": "left side", "middle": "middle", "right": "right side",
       "inside": "inside runs", "tackle": "off-tackle runs", "outside": "outside runs"}
matchups = []; sched = []
if next_week:
    wk = games[(games.week == next_week) & games.result.isna()]
    opp = {}
    for _, gm in wk.iterrows():
        opp[gm.home_team] = gm.away_team; opp[gm.away_team] = gm.home_team
        sched.append({"away": gm.away_team, "home": gm.home_team, "day": gm.gameday, "time": gm.gametime})
    pbyid = {p["id"]: p for p in players}
    for p in players:
        o = opp.get(p["team"])
        if not o or o not in teams: continue
        D = teams[o]; notes = []; edge = 0.0
        if "recv" in p and p["recv"]["n"] >= 15 and p["pos"] in ("WR", "TE", "RB"):
            R = p["recv"]
            for zn in ["short", "mid", "deep"]:
                sh = (R["depth"].get(zn) or 0) / 100; dv = D["dDepth"][zn]
                if dv["epa"] is None: continue
                diff = dv["epa"] - lg["depth"][zn]; edge += sh * diff
                if sh >= 0.25 and dv["rk"] >= 24: notes.append(f"+ {int(sh*100)}% of targets {LBL[zn]} · {o} #{dv['rk']} vs {LBL[zn]} passes")
                if sh >= 0.25 and dv["rk"] <= 8: notes.append(f"− {int(sh*100)}% of targets {LBL[zn]} · {o} #{dv['rk']} vs {LBL[zn]} passes")
            for zn in ["left", "middle", "right"]:
                sh = (R["side"].get(zn) or 0) / 100; dv = D["dSide"][zn]
                if dv["epa"] is None: continue
                edge += 0.5 * sh * (dv["epa"] - lg["side"][zn])
                if sh >= 0.4 and dv["rk"] >= 26: notes.append(f"+ {int(sh*100)}% of targets to the {LBL[zn]} · {o} #{dv['rk']} there")
            dv = D["dPos"].get(p["pos"])
            if dv and dv.get("epa") is not None:
                edge += 0.5 * (dv["epa"] - lg["pos"][p["pos"]])
                if dv["rk"] >= 26: notes.append(f"+ {o} #{dv['rk']} vs {p['pos']}s")
                if dv["rk"] <= 5: notes.append(f"− {o} #{dv['rk']} vs {p['pos']}s")
            kind = "rec"
        elif "rush" in p and p["rush"]["n"] >= 30:
            R = p["rush"]
            for zn in ["inside", "tackle", "outside"]:
                sh = (R["zone"].get(zn) or 0) / 100; dv = D["dRun"][zn]
                if dv["epa"] is None: continue
                edge += sh * (dv["epa"] - lg["run"][zn])
                if sh >= 0.3 and dv["rk"] >= 24: notes.append(f"+ {int(sh*100)}% {LBL[zn]} · {o} #{dv['rk']} vs them")
                if sh >= 0.3 and dv["rk"] <= 8: notes.append(f"− {int(sh*100)}% {LBL[zn]} · {o} #{dv['rk']} vs them")
            kind = "rush"
        elif "qb" in p and p["qb"]["n"] >= 80:
            Q = p["qb"]
            for zn in ["short", "mid", "deep"]:
                sh = ((Q["depth"].get(zn) or {}).get("pct") or 0) / 100; dv = D["dDepth"][zn]
                if dv["epa"] is None: continue
                edge += sh * (dv["epa"] - lg["depth"][zn])
                if zn == "deep" and sh >= 0.12 and dv["rk"] >= 24: notes.append(f"+ Throws deep {int(sh*100)}% · {o} #{dv['rk']} vs deep passes")
                if zn == "deep" and sh >= 0.12 and dv["rk"] <= 8: notes.append(f"− Throws deep {int(sh*100)}% · {o} #{dv['rk']} vs deep passes")
            if D["passDef_rk"] >= 26: notes.append(f"+ {o} has the #{D['passDef_rk']} pass defense")
            if D["passDef_rk"] <= 5: notes.append(f"− {o} has the #{D['passDef_rk']} pass defense")
            kind = "qb"
        else:
            continue
        matchups.append({"id": p["id"], "name": p["name"], "pos": p["pos"], "team": p["team"], "opp": o,
                         "kind": kind, "edge": r(edge * 100, 1), "notes": notes})
    matchups.sort(key=lambda m: -(m["edge"] or 0))


# ---------------- prop picks for next week ----------------
from math import erf, sqrt, exp
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
pw["anytd"] = pw.rushing_tds + pw.receiving_tds
PSTAT = {  # stat: (column, label, ladder, positions)
    "pyd": ("passing_yards", "Pass Yds", [200, 225, 250, 275, 300], ["QB"]),
    "ptd": ("passing_tds", "Pass TDs", [1, 2, 3], ["QB"]),
    "qbryd": ("rushing_yards", "Rush Yds", [20, 25, 30, 40, 50], ["QB"]),
    "ryd": ("rushing_yards", "Rush Yds", [40, 50, 60, 70, 80, 90, 100], ["RB"]),
    "car": ("carries", "Carries", [12, 14, 16, 18, 20], ["RB"]),
    "rec": ("receptions", "Receptions", [3, 4, 5, 6, 7, 8], ["RB", "WR", "TE"]),
    "recyd": ("receiving_yards", "Rec Yds", [30, 40, 50, 60, 70, 80, 90, 100], ["RB", "WR", "TE"]),
    "anytd": ("anytd", "Anytime TD", [1], ["RB", "WR", "TE"]),
}
# what each defense allows per game to each position group
allow = pw.groupby(["opponent_team", "position_group", "week"])[["passing_yards", "passing_tds", "rushing_yards", "carries",
        "receptions", "receiving_yards", "anytd"]].sum().groupby(["opponent_team", "position_group"]).mean()
lg_allow = allow.groupby("position_group").mean()
def allow_rank(pos, col, team):
    try: s_ = allow.xs(pos, level="position_group")[col].sort_values()
    except KeyError: return None
    return int(list(s_.index).index(team)) + 1 if team in s_.index else None
team_last = pw.groupby("team").week.max().to_dict()
props = []
if next_week:
    opp_of = {}; home_of = {}
    for g_ in sched: opp_of[g_["home"]] = g_["away"]; opp_of[g_["away"]] = g_["home"]; home_of[g_["home"]] = True; home_of[g_["away"]] = False
    mu_by = {m["id"]: m for m in matchups}
    PL = {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs"}
    for pid, g in pw.groupby("player_id"):
        L = latest.loc[pid]; pos = L.position_group; tm_ = L.team
        if pos not in ("QB", "RB", "WR", "TE") or tm_ not in opp_of: continue
        g = g[g.team == tm_].sort_values("week")
        if len(g) < 3 or g.week.max() != team_last.get(tm_): continue  # needs 3+ games and played team's last game
        if pos == "QB" and g.attempts.mean() < 15: continue
        o = opp_of[tm_]
        for sk, (col, lab, ladder, poss) in PSTAT.items():
            if pos not in poss: continue
            v = g[col].astype(float).values; n = len(v)
            mean = v.mean(); last3 = v[-3:].mean()
            if mean <= 0: continue
            base = 0.6 * mean + 0.4 * last3
            try: oa = allow.loc[(o, pos), col]; la = lg_allow.loc[pos, col]
            except KeyError: oa = la = None
            fac = 1.0
            if oa is not None and la: fac = max(0.82, min(1.18, 1 + 0.5 * (oa / la - 1)))
            proj = base * fac
            if sk == "anytd":
                lam = proj; prob = 1 - exp(-lam); hits = int((v >= 1).sum()); T = 1
                if prob < 0.5: continue
                best = (T, prob, hits)
            else:
                sd = max(v.std(ddof=1) if n > 1 else 0, 0.3 * mean, 1.0)
                best = None
                for T in ladder:
                    if T < 0.7 * proj: continue  # skip trivial lines
                    pn = 1 - Phi((T - 0.5 - proj) / sd)
                    hits = int((v >= T).sum()); pe = (hits + 1) / (n + 2)
                    prob = 0.6 * pn + 0.4 * pe
                    prob = 0.5 + (prob - 0.5) * min(1.0, n / 6)
                    if prob >= 0.64: best = (T, prob, hits)
                if not best: continue
            T, prob, hits = best
            prob = min(prob, 0.88)
            rk_ = allow_rank(pos, col, o)
            reasons = []
            unit = lab if "TD" in lab else lab.lower()
            if sk == "anytd":
                reasons.append(f"{int(v.sum())} total TDs in {n} games ({hits} of {n} games with a TD)")
            else:
                reasons.append(f"Averaging {mean:.1f} {unit}/game (last 3: {last3:.1f})")
                reasons.append(f"Hit {T}+ in {hits} of {n} games")
            if sk == "ryd": reasons.append(f"Getting {g.carries.mean():.1f} carries/game")
            if sk in ("rec", "recyd"):
                ts = g.target_share.dropna().mean() if "target_share" in g else None
                reasons.append(f"{g.targets.mean():.1f} targets/game" + (f" ({ts*100:.0f}% target share)" if ts and ts == ts else ""))
            if oa is not None and rk_:
                worst = rk_ >= 22; best_ = rk_ <= 10
                reasons.append(f"{o} allows {oa:.1f} {unit}/game to {PL[pos]} (#{rk_} of 32{', a weak spot' if worst else ', a tough spot' if best_ else ''})")
            T_o = teams.get(tm_, {}); T_d = teams.get(o, {})
            if sk in ("ryd", "car", "qbryd") and T_d.get("rushDef_rk"): reasons.append(f"{o} run defense ranks #{T_d['rushDef_rk']}; {tm_} run offense ranks #{T_o.get('rushOff_rk')}")
            if sk in ("pyd", "ptd", "rec", "recyd") and T_d.get("passDef_rk"): reasons.append(f"{o} pass defense ranks #{T_d['passDef_rk']}; {tm_} pass offense ranks #{T_o.get('passOff_rk')}")
            mu = mu_by.get(pid); risk = None
            if mu:
                pos_notes = [x[2:] for x in mu["notes"] if x.startswith("+")]
                neg_notes = [x[2:] for x in mu["notes"] if x.startswith("−")]
                if pos_notes: reasons.append(pos_notes[0])
                if neg_notes: risk = neg_notes[0]
            if risk is None and rk_ and rk_ <= 8: risk = f"{o} is one of the stingier defenses vs {PL[pos]}"
            if risk is None and last3 < 0.8 * mean: risk = f"Trending down lately ({last3:.1f} over the last 3)"
            props.append({"id": pid, "name": L.player_display_name, "pos": L.position, "team": tm_, "opp": o,
                          "home": home_of[tm_], "stat": sk, "label": lab, "line": T, "proj": r(proj, 1),
                          "prob": r(prob * 100, 0), "hits": hits, "n": n, "reasons": reasons, "risk": risk})
    # keep the best 2 props per player
    props.sort(key=lambda x: -x["prob"])
    seen = {}; kept = []
    for p_ in props:
        c_ = seen.get(p_["id"], 0)
        if c_ < 2: kept.append(p_); seen[p_["id"]] = c_ + 1
    props = kept

# ---------------- full projections for next week ----------------
projections = {}
if next_week:
    PCOLS = {"QB": [("pyd","passing_yards"),("ptd","passing_tds"),("int","passing_interceptions"),("car","carries"),("ryd","rushing_yards")],
             "RB": [("car","carries"),("ryd","rushing_yards"),("rtd","rushing_tds"),("tgt","targets"),("rec","receptions"),("recyd","receiving_yards")],
             "WR": [("tgt","targets"),("rec","receptions"),("recyd","receiving_yards"),("rectd","receiving_tds"),("ryd","rushing_yards")],
             "TE": [("tgt","targets"),("rec","receptions"),("recyd","receiving_yards"),("rectd","receiving_tds")]}
    pinfo = {p["id"]: p for p in players}
    # defenders to watch on each team
    watch = {}
    for t in TEAMS:
        roster_ = [p for p in players if p["team"] == t]
        dbs = sorted([p for p in roster_ if p["grp"] == "DB"], key=lambda p: -(p["tot"]["dint"] * 3 + p["tot"]["pd"]))[:3]
        rush_ = sorted([p for p in roster_ if p["grp"] in ("DL", "LB")], key=lambda p: -(p["tot"]["sacks"] * 3 + p["tot"]["qbh"]))[:2]
        tack = sorted([p for p in roster_ if p["grp"] in ("LB", "DL")], key=lambda p: -(p["tot"]["tfl"] * 3 + p["tot"]["tkl"] + p["tot"]["ast"] * 0.5))[:2]
        fm = lambda p, kind: {"id": p["id"], "name": p["name"], "pos": p["pos"],
            "line": (f"{int(p['tot']['dint'])} INT · {int(p['tot']['pd'])} PD" if kind == "cov" else
                     f"{p['tot']['sacks']:g} sacks · {int(p['tot']['qbh'])} QB hits" if kind == "rush" else
                     f"{int(p['tot']['tkl'] + p['tot']['ast'])} tkl · {int(p['tot']['tfl'])} TFL")}
        watch[t] = {"cov": [fm(p, "cov") for p in dbs], "rush": [fm(p, "rush") for p in rush_], "run": [fm(p, "run") for p in tack]}
    mu_by = {m["id"]: m for m in matchups}
    for pid, g in pw.groupby("player_id"):
        L = latest.loc[pid]; pos = L.position_group; tm_ = L.team
        if pos not in PCOLS or tm_ not in opp_of: continue
        g = g[g.team == tm_].sort_values("week")
        if len(g) < 2 or g.week.max() != team_last.get(tm_): continue
        if pos == "QB" and g.attempts.mean() < 15: continue
        if pos != "QB" and (g.targets.mean() + g.carries.mean()) < 2.5: continue
        o = opp_of[tm_]; mu = mu_by.get(pid); edge = (mu or {}).get("edge") or 0
        zf = 1 + max(-0.12, min(0.12, edge / 400))  # strengths-vs-weaknesses tilt
        stats = {}
        for k, col in PCOLS[pos]:
            v = g[col].astype(float).values
            base = 0.6 * v.mean() + 0.4 * v[-3:].mean()
            if base < 0.05 and k in ("ryd", "car"): continue
            try: fac = max(0.82, min(1.18, 1 + 0.5 * (allow.loc[(o, pos), col] / lg_allow.loc[pos, col] - 1)))
            except Exception: fac = 1.0
            f2 = fac * (zf if k not in ("car", "tgt", "int") else 1 + (zf - 1) * 0.4)
            pj = base * max(0.75, min(1.25, f2))
            sd = max(v.std(ddof=1) if len(v) > 1 else 0, 0.3 * max(base, 0.01))
            stats[k] = [r(pj, 1), r(max(0, pj - 0.67 * sd), 0 if pj > 5 else 1), r(pj + 0.67 * sd, 0 if pj > 5 else 1)]
        # strengths & weaknesses
        P = pinfo.get(pid, {}); st = []; wk = []
        R = P.get("recv"); RU = P.get("rush"); Q = P.get("qb")
        if R and pos != "QB" and R["n"] >= 15:
            if (R["depth"].get("deep") or 0) >= 25: st.append(f"Deep threat: {int(R['depth']['deep'])}% of targets 20+ yards")
            if (R["depth"].get("short") or 0) >= 60: st.append(f"Short-area target: {int(R['depth']['short'])}% of targets under 10 yards")
            if (R.get("yac") or 0) >= 5.5: st.append(f"Yards after catch: {R['yac']} per catch")
            if (R.get("catch") or 0) >= 75: st.append(f"Sure hands: {int(R['catch'])}% catch rate")
            if (R.get("catch") or 100) < 58: wk.append(f"Low catch rate: {int(R['catch'])}%")
        if RU and pos in ("RB", "QB") and RU["n"] >= 20:
            z = max(RU["zone"], key=lambda k_: RU["zone"][k_] or 0)
            st.append(f"Runs mostly {({'inside':'inside','tackle':'off tackle','outside':'outside'})[z]} ({int(RU['zone'][z])}%)")
            if (RU.get("explosive") or 0) >= 14: st.append(f"Big-play runner: {int(RU['explosive'])}% of runs go 10+")
            if (RU.get("stuff") or 0) >= 22: wk.append(f"Gets stuffed on {int(RU['stuff'])}% of runs")
        if Q:
            dz = Q["depth"]
            if (dz.get("deep", {}).get("pct") or 0) >= 13: st.append(f"Throws deep {int(dz['deep']['pct'])}% of the time ({int(dz['deep'].get('cmp') or 0)}% complete)")
            if (dz.get("deep", {}).get("cmp") or 100) < 30: wk.append(f"Struggles deep: {int(dz['deep']['cmp'])}% complete on 20+ yd throws")
            if P["tot"]["int"] / max(P["gp"], 1) >= 1: wk.append(f"Turnover prone: {int(P['tot']['int'])} INTs")
            if P["tot"]["sk"] / max(P["gp"], 1) >= 3: wk.append(f"Sacked {P['tot']['sk'] / P['gp']:.1f} times a game")
        dw = [n[2:] for n in (mu or {}).get("notes", []) if n.startswith("+")]
        ds = [n[2:] for n in (mu or {}).get("notes", []) if n.startswith("−")]
        projections[pid] = {"opp": o, "home": home_of[tm_], "edge": edge, "stats": stats,
                            "strengths": st[:4], "weaknesses": wk[:3], "oppWeak": dw[:4], "oppStrong": ds[:3],
                            "watch": watch.get(o, {}).get("run" if pos == "RB" else "rush" if pos == "QB" else "cov", [])}

out = {"season": SEASON, "lastWeek": last_week, "nextWeek": next_week,
       "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
       "keys": KEYS, "players": players, "teams": teams, "matchups": matchups, "schedule": sched, "props": props, "proj": projections,
       "lg": {c: {k: r(v, 3) for k, v in d.items()} for c, d in lg.items()}}
js = json.dumps(out, separators=(",", ":"), allow_nan=False, default=str)
with open(os.path.join(OUT, "data.json"), "w") as f: f.write(js)
print(f"players={len(players)} teams={len(teams)} matchups={len(matchups)} lastWeek={last_week} nextWeek={next_week} bytes={len(js)}")

# embed data into index.html (if present) so the page needs no separate fetch
import re
ip = os.path.join(OUT, "index.html")
if os.path.exists(ip):
    html = open(ip, encoding="utf-8").read()
    safe = js.replace("</", "<\\/")
    html, n = re.subn(r'(<script id="gl-data" type="application/json">)(.*?)(</script>)', lambda m: m.group(1) + safe + m.group(3), html, count=1, flags=re.S)
    open(ip, "w", encoding="utf-8").write(html)
    print("embedded data into index.html" if n else "WARNING: gl-data tag not found in index.html")
