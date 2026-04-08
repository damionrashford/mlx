#!/usr/bin/env python3
# MCP experiment tracking server — Python stdlib only, no pip installs.
# JSON-RPC 2.0 over stdio. Reads/writes results.tsv from CWD.

import sys
import json
import os
import csv
import io

TSV_FILENAME = "results.tsv"
COLUMNS = ["id", "metric", "val_score", "test_score", "memory_mb", "status", "description"]


def get_tsv_path():
    data_dir = os.environ.get("MLX_DATA_DIR", "")
    cwd = os.getcwd()
    # Prefer CWD results.tsv (active project), fallback to MLX_DATA_DIR
    cwd_tsv = os.path.join(cwd, TSV_FILENAME)
    if os.path.exists(cwd_tsv):
        return cwd_tsv
    if data_dir and os.path.exists(os.path.join(data_dir, TSV_FILENAME)):
        return os.path.join(data_dir, TSV_FILENAME)
    return cwd_tsv  # default to CWD even if not yet created


def read_experiments():
    path = get_tsv_path()
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def write_row(row_dict):
    path = get_tsv_path()
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


# --- Tool implementations ---

def tool_get_best_run(params):
    rows = read_experiments()
    keep_rows = [r for r in rows if r.get("status", "").upper() == "KEEP"]
    if not keep_rows:
        return {"result": None, "message": "No KEEP experiments found in results.tsv"}
    best = max(keep_rows, key=lambda r: float(r.get("val_score", 0) or 0))
    return {"result": best}


def tool_list_experiments(params):
    rows = read_experiments()
    return {"experiments": rows, "count": len(rows)}


def tool_compare_experiments(params):
    id1 = params.get("id1")
    id2 = params.get("id2")
    rows = read_experiments()
    by_id = {r.get("id"): r for r in rows}
    r1 = by_id.get(str(id1))
    r2 = by_id.get(str(id2))
    if not r1:
        return {"error": f"Experiment '{id1}' not found"}
    if not r2:
        return {"error": f"Experiment '{id2}' not found"}
    comparison = {}
    for col in COLUMNS:
        comparison[col] = {str(id1): r1.get(col, ""), str(id2): r2.get(col, "")}
    return {"comparison": comparison}


def tool_add_experiment(params):
    required = ["id", "metric", "val_score", "status", "description"]
    for key in required:
        if key not in params:
            return {"error": f"Missing required field: {key}"}
    row = {
        "id": params["id"],
        "metric": params["metric"],
        "val_score": str(params["val_score"]),
        "test_score": str(params.get("test_score", "")),
        "memory_mb": str(params.get("memory_mb", "")),
        "status": str(params["status"]).upper(),
        "description": params["description"],
    }
    write_row(row)
    return {"result": "ok", "row": row}


def tool_get_experiment_summary(params):
    rows = read_experiments()
    total = len(rows)
    keep = sum(1 for r in rows if r.get("status", "").upper() == "KEEP")
    discard = sum(1 for r in rows if r.get("status", "").upper() == "DISCARD")
    crash = sum(1 for r in rows if r.get("status", "").upper() == "CRASH")
    keep_rows = [r for r in rows if r.get("status", "").upper() == "KEEP"]
    best_val = max((float(r.get("val_score", 0) or 0) for r in keep_rows), default=None)
    return {
        "total": total,
        "keep": keep,
        "discard": discard,
        "crash": crash,
        "best_val_score": best_val,
    }


TOOLS = {
    "get_best_run": {
        "description": "Returns the highest val_score KEEP row from results.tsv",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "fn": tool_get_best_run,
    },
    "list_experiments": {
        "description": "Returns all experiment rows from results.tsv as structured JSON",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "fn": tool_list_experiments,
    },
    "compare_experiments": {
        "description": "Side-by-side diff of two experiment rows by id",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id1": {"type": "string", "description": "First experiment ID"},
                "id2": {"type": "string", "description": "Second experiment ID"},
            },
            "required": ["id1", "id2"],
        },
        "fn": tool_compare_experiments,
    },
    "add_experiment": {
        "description": "Appends a new experiment row to results.tsv",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "metric": {"type": "string"},
                "val_score": {"type": "number"},
                "test_score": {"type": "number"},
                "memory_mb": {"type": "number"},
                "status": {"type": "string", "enum": ["KEEP", "DISCARD", "CRASH"]},
                "description": {"type": "string"},
            },
            "required": ["id", "metric", "val_score", "status", "description"],
        },
        "fn": tool_add_experiment,
    },
    "get_experiment_summary": {
        "description": "Returns total, KEEP, DISCARD, CRASH counts and best val_score",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "fn": tool_get_experiment_summary,
    },
}


def send(obj):
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def handle_request(req):
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mlx-experiments", "version": "1.0.0"},
            },
        })
    elif method == "tools/list":
        tools_list = []
        for name, t in TOOLS.items():
            tools_list.append({
                "name": name,
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            })
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}})
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})
        if tool_name not in TOOLS:
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
            })
            return
        try:
            result = TOOLS[tool_name]["fn"](tool_params)
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                },
            })
        except Exception as e:
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            })
    elif method == "notifications/initialized":
        pass  # no response needed
    else:
        if req_id is not None:
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle_request(req)


if __name__ == "__main__":
    main()
