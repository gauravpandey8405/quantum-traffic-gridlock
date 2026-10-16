import numpy as np
import json

def parse_scenario(sc):
    """
    Maps both standard flat keys and the new Scenario H dictionary keys 
    so your existing functions work seamlessly without changes.
    """
    # 1. Map capacity parameters
    cap = sc.get("cap", sc.get("service_capacity", 5))
    S = sc.get("S", sc.get("storage_capacity", 7))
    wo = sc.get("wo", sc.get("overflow_weight", 4))
    ws = sc.get("ws", sc.get("switch_weight", 0.5))
    
    # 2. Map link configuration (handling both strings and dictionaries)
    link_raw = sc.get("link", "EW")
    if isinstance(link_raw, dict):
        link = link_raw.get("phase", "NS")
        fr = link_raw.get("fraction", 0.6)
    else:
        link = link_raw
        fr = sc.get("fr", 0.6)

    parsed = {
        "cap": cap,
        "S": S,
        "wo": wo,
        "ws": ws,
        "link": link,
        "fr": fr
    }
    
    # 3. Map arrivals (supporting both flat keys and nested 'arrivals' dictionaries)
    arrivals = sc.get("arrivals", {})
    if arrivals:
        parsed["A_NS"] = arrivals.get("A", {}).get("NS", [3, 6, 6, 3])
        parsed["A_EW"] = arrivals.get("A", {}).get("EW", [5, 1, 1, 2])
        b_data = arrivals.get("B", {"NS": [3, 5, 4, 3], "EW": [2, 6, 4, 6]})
        parsed["B_NS"] = b_data.get("NS", [3, 5, 4, 3])
        parsed["B_EW"] = b_data.get("EW", [2, 6, 4, 6])
    else:
        parsed["A_NS"] = sc.get("A_NS", [3, 6, 6, 3])
        parsed["A_EW"] = sc.get("A_EW", [5, 1, 1, 2])
        parsed["B_NS"] = sc.get("B_NS", [3, 5, 4, 3])
        parsed["B_EW"] = sc.get("B_EW", [2, 6, 4, 6])
        
    return parsed

def official_cost(bits, sc):
    sc = parse_scenario(sc)
    cap, S, wo, ws = sc["cap"], sc["S"], sc["wo"], sc["ws"]
    link, fr = sc["link"], sc["fr"]
    A, B = bits[:4], bits[4:]
    arr = {("A","NS"): sc["A_NS"], ("A","EW"): sc["A_EW"],
           ("B","NS"): sc["B_NS"], ("B","EW"): sc["B_EW"]}
    q = {k: 0 for k in arr}
    delay = ovf = carry = 0
    
    for t in range(4):
        for k in arr:
            q[k] += arr[k][t]
        q[("B", link)] += carry
        dep_link = 0
        for I, x in (("A", A[t]), ("B", B[t])):
            g = "NS" if x == 1 else "EW"
            s = min(q[(I, g)], cap)
            q[(I, g)] -= s
            if I == "A" and g == link:
                dep_link = s
        delay += sum(q.values())
        ovf += sum(max(0, v - S) for v in q.values())
        carry = int(np.floor(fr * dep_link + 0.5))
        
    sw = sum(A[i] != A[i+1] for i in range(3)) + sum(B[i] != B[i+1] for i in range(3))
    return float(delay + wo * ovf + ws * sw)

def official_costs(sc):
    sc = parse_scenario(sc)
    costs = np.zeros(256)
    for i in range(256):
        bits = [(i >> k) & 1 for k in range(8)]
        costs[i] = official_cost(bits, sc)
    return costs

def elite_set(costs, n=13):
    sorted_indices = sorted(range(len(costs)), key=lambda idx: (costs[idx], idx))
    return sorted_indices[:n]

def build_qubo(sc, alpha=0.1):
    sc = parse_scenario(sc)
    Q = np.zeros((8, 8))
    offset = 0.0
    
    ws = sc["ws"]
    for i in range(3):
        Q[i, i] += ws
        Q[i+1, i+1] += ws
        Q[i, i+1] -= ws
        
        Q[i+4, i+4] += ws
        Q[i+5, i+5] += ws
        Q[i+4, i+5] -= ws

    cap = sc["cap"]
    for t in range(4):
        Q[t, t] -= alpha * cap * (t + 1)
        Q[t+4, t+4] -= alpha * cap * (t + 1)
        offset += alpha * sum(sc["A_NS"])**2 + alpha * sum(sc["B_NS"])**2

    return Q, float(offset)

def export_outputs(sc, filename_qubo="qubo.json", filename_schedule="schedule.json"):
    Q, offset = build_qubo(sc)
    qubo_data = {
        "Q": Q.tolist(),
        "offset": offset
    }
    with open(filename_qubo, "w") as f:
        json.dump(qubo_data, f, indent=4)
        
    costs = official_costs(sc)
    elite = elite_set(costs, 13)
    schedule_data = {
        "best_index": int(elite[0]),
        "best_cost": float(costs[elite[0]]),
        "random_average": float(costs.mean()),
        "elite_indices": elite
    }
    with open(filename_schedule, "w") as f:
        json.dump(schedule_data, f, indent=4)