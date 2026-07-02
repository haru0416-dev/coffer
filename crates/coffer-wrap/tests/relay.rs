//! End-to-end relay tests over in-memory duplex pipes: a scripted fake downstream MCP
//! server on one side, a test client on the other, the real `run_relay` in between.

use coffer_wrap::{HandleStore, WrapConfig, run_relay};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

/// A synthetic large JSON-array payload (well over the offload threshold).
fn big_payload(rows: usize) -> String {
    let items: Vec<Value> = (0..rows)
        .map(|i| {
            json!({
                "name": format!("pod-{i}"),
                "status": if i % 10 == 0 { "Error" } else { "Running" },
                "restarts": i % 7
            })
        })
        .collect();
    Value::Array(items).to_string()
}

/// Scripted downstream: answers initialize / tools/list / tools/call like a real stdio
/// MCP server would, one JSON-RPC message per line.
async fn fake_downstream<R, W>(reader: R, mut writer: W, tools: Vec<Value>, payload: String)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let msg: Value = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let (Some(method), Some(id)) = (
            msg.get("method").and_then(Value::as_str),
            msg.get("id").cloned(),
        ) else {
            continue; // notification or response — nothing to answer
        };
        let result = match method {
            "initialize" => json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-downstream", "version": "0.0.0"}
            }),
            "tools/list" => json!({"tools": tools}),
            "tools/call" => {
                let name = msg
                    .pointer("/params/name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let text = match name {
                    "big" => payload.clone(),
                    "coffer_digest" => "downstream-digest-ok".to_string(),
                    _ => "ok".to_string(),
                };
                json!({"content": [{"type": "text", "text": text}], "isError": false})
            }
            _ => json!({}),
        };
        let resp = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
        if writer.write_all(resp.as_bytes()).await.is_err() {
            break;
        }
        let _ = writer.write_all(b"\n").await;
        let _ = writer.flush().await;
    }
}

struct Client {
    writer: tokio::io::WriteHalf<tokio::io::DuplexStream>,
    lines: tokio::io::Lines<BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>>,
    next_id: u64,
}

impl Client {
    /// Send a request and await the response bearing the same id.
    async fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.writer
            .write_all(req.to_string().as_bytes())
            .await
            .unwrap();
        self.writer.write_all(b"\n").await.unwrap();
        self.writer.flush().await.unwrap();
        loop {
            let line = tokio::time::timeout(Duration::from_secs(10), self.lines.next_line())
                .await
                .expect("timed out waiting for a response")
                .unwrap()
                .expect("relay closed unexpectedly");
            let msg: Value = serde_json::from_str(&line).unwrap();
            if msg.get("id").and_then(Value::as_u64) == Some(id) && msg.get("method").is_none() {
                return msg;
            }
        }
    }

    async fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        self.request("tools/call", json!({"name": name, "arguments": arguments}))
            .await
    }
}

/// Wire up fake-downstream <-> relay <-> client and hand back the client.
fn spawn_stack(tools: Vec<Value>, payload: String) -> Client {
    let (client_end, relay_client_end) = tokio::io::duplex(4 * 1024 * 1024);
    let (relay_down_end, down_end) = tokio::io::duplex(4 * 1024 * 1024);
    let (rc_read, rc_write) = tokio::io::split(relay_client_end);
    let (rd_read, rd_write) = tokio::io::split(relay_down_end);
    let (d_read, d_write) = tokio::io::split(down_end);

    tokio::spawn(fake_downstream(d_read, d_write, tools, payload));
    tokio::spawn(run_relay(
        rc_read,
        rc_write,
        rd_write,
        rd_read,
        HandleStore::Memory(coffer_cas::MemoryCas::new()),
        WrapConfig::default(),
    ));

    let (c_read, c_write) = tokio::io::split(client_end);
    Client {
        writer: c_write,
        lines: BufReader::new(c_read).lines(),
        next_id: 0,
    }
}

fn downstream_tool(name: &str) -> Value {
    json!({"name": name, "description": "a downstream tool", "inputSchema": {"type": "object"}})
}

fn extract_handle(card: &str) -> String {
    let line = card
        .lines()
        .find(|l| l.starts_with("handle: "))
        .expect("fact card must carry a handle line");
    line.trim_start_matches("handle: ").trim().to_string()
}

#[tokio::test]
async fn initialize_passes_through_verbatim() {
    let mut client = spawn_stack(vec![downstream_tool("big")], big_payload(10));
    let resp = client
        .request("initialize", json!({"protocolVersion": "2025-06-18"}))
        .await;
    assert_eq!(
        resp.pointer("/result/serverInfo/name")
            .and_then(Value::as_str),
        Some("fake-downstream")
    );
}

#[tokio::test]
async fn tools_list_gains_injected_tools() {
    let mut client = spawn_stack(vec![downstream_tool("big")], big_payload(10));
    let resp = client.request("tools/list", json!({})).await;
    let names: Vec<&str> = resp
        .pointer("/result/tools")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|t| t.get("name").unwrap().as_str().unwrap())
        .collect();
    assert!(names.contains(&"big"), "{names:?}");
    for injected in [
        "coffer_describe",
        "coffer_digest",
        "coffer_aggregate",
        "coffer_search",
        "coffer_lines",
        "coffer_retrieve",
    ] {
        assert!(names.contains(&injected), "missing {injected}: {names:?}");
    }
}

#[tokio::test]
async fn small_results_pass_through_verbatim() {
    let mut client = spawn_stack(vec![downstream_tool("small")], big_payload(10));
    let resp = client.call_tool("small", json!({})).await;
    assert_eq!(
        resp.pointer("/result/content/0/text")
            .and_then(Value::as_str),
        Some("ok")
    );
}

#[tokio::test]
async fn large_result_offloads_and_round_trips_byte_exact() {
    let payload = big_payload(3000);
    let mut client = spawn_stack(vec![downstream_tool("big")], payload.clone());

    // 1. The oversized result is replaced by a fact card carrying a handle.
    let resp = client.call_tool("big", json!({})).await;
    let card = resp
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .unwrap();
    assert!(
        card.starts_with("[coffer-wrap] tool result offloaded"),
        "{card}"
    );
    assert!(
        !card.contains("pod-2999"),
        "original rows must not leak into the card"
    );
    assert!(
        card.contains("3000 records"),
        "describe should count rows: {card}"
    );
    let handle = extract_handle(card);
    assert_eq!(handle.len(), 64, "full sha256 handle expected: {handle}");

    // 2. retrieve returns the original bytes exactly.
    let resp = client
        .call_tool(
            "coffer_retrieve",
            json!({"handle": handle, "len": 4 * 1024 * 1024}),
        )
        .await;
    // The configured cap (1 MiB) exceeds the payload, so the full window comes back raw.
    assert_eq!(
        resp.pointer("/result/content/0/text")
            .and_then(Value::as_str),
        Some(payload.as_str()),
        "retrieve must round-trip byte-exact"
    );

    // 3. aggregate answers exactly, with provenance.
    let resp = client
        .call_tool(
            "coffer_aggregate",
            json!({
                "handle": handle,
                "where": [{"field": "status", "op": "eq", "value": "Error"}],
                "agg": "count"
            }),
        )
        .await;
    let text = resp
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .unwrap();
    assert_eq!(resp.pointer("/result/isError"), Some(&json!(false)));
    assert!(
        text.contains("300"),
        "3000 rows, every 10th is Error: {text}"
    );
    assert!(text.contains("provenance"), "{text}");

    // 4. describe sees the schema.
    let resp = client
        .call_tool("coffer_describe", json!({"handle": handle}))
        .await;
    let text = resp
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .unwrap();
    assert!(text.contains("restarts"), "{text}");
}

#[tokio::test]
async fn unknown_aggregate_op_is_refused() {
    let payload = big_payload(3000);
    let mut client = spawn_stack(vec![downstream_tool("big")], payload);
    let resp = client.call_tool("big", json!({})).await;
    let handle = extract_handle(
        resp.pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .unwrap(),
    );
    let resp = client
        .call_tool(
            "coffer_aggregate",
            json!({
                "handle": handle,
                "where": [{"field": "restarts", "op": "gte", "value": 1}],
                "agg": "count"
            }),
        )
        .await;
    assert_eq!(resp.pointer("/result/isError"), Some(&json!(true)));
    let text = resp
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .unwrap();
    assert!(text.contains("gte"), "must name the rejected op: {text}");
}

#[tokio::test]
async fn colliding_downstream_tool_is_not_shadowed() {
    // Downstream has its own `coffer_digest`; ours must rename and routing must forward.
    let mut client = spawn_stack(
        vec![downstream_tool("big"), downstream_tool("coffer_digest")],
        big_payload(10),
    );
    let resp = client.request("tools/list", json!({})).await;
    let names: Vec<&str> = resp
        .pointer("/result/tools")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|t| t.get("name").unwrap().as_str().unwrap())
        .collect();
    assert_eq!(
        names.iter().filter(|n| **n == "coffer_digest").count(),
        1,
        "{names:?}"
    );
    assert!(names.contains(&"wrap_coffer_digest"), "{names:?}");

    // Calling the downstream's name reaches the downstream, not our handler.
    let resp = client.call_tool("coffer_digest", json!({})).await;
    assert_eq!(
        resp.pointer("/result/content/0/text")
            .and_then(Value::as_str),
        Some("downstream-digest-ok")
    );
}

#[tokio::test]
async fn unknown_handle_is_a_clean_in_band_error() {
    let mut client = spawn_stack(vec![downstream_tool("big")], big_payload(10));
    let resp = client
        .call_tool("coffer_describe", json!({"handle": "ab".repeat(32)}))
        .await;
    assert_eq!(resp.pointer("/result/isError"), Some(&json!(true)));
    let text = resp
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)
        .unwrap();
    assert!(text.contains("unknown handle"), "{text}");
}
