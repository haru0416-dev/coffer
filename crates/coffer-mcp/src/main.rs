//! coffer MCP server: hold large agent tool-output **server-side** and let the model
//! direct coffer at it via opaque handles — exact `digest` aggregates, predicate `query`,
//! targeted `search`/`lines`, and byte-exact `retrieve` — so huge tool-output never enters the
//! model's context.

use std::path::Path;

use coffer_core::{Predicate, compress_json_where, query_subset};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{ErrorData, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

mod limits;
mod render;
mod run;
mod store;

use limits::*;
use render::*;
use run::*;
use store::*;

/// Build a conjunction of typed predicates from the wire form. Shared by coffer_select / coffer_aggregate.
fn predicates_from_args(args: &[PredicateArg]) -> Vec<Predicate> {
    args.iter()
        .map(|p| Predicate {
            field: p.field.clone(),
            op: parse_op(&p.op),
            value: parse_value_arg(&p.value),
        })
        .collect()
}

pub(crate) fn unfold_shared_cas_result(
    path: impl AsRef<Path>,
    args: &UnfoldArgs,
    limits: RetrieveLimits,
) -> CallToolResult {
    match coffer_cas::read_blob(path, &args.hash) {
        Ok(Some(bytes)) => match render_retrieved_bytes(
            &bytes,
            args.start,
            args.max_bytes,
            args.full.unwrap_or(false),
            limits,
        ) {
            Ok(text) => CallToolResult::success(vec![Content::text(text)]),
            Err(e) => CallToolResult::error(vec![Content::text(e)]),
        },
        Ok(None) => CallToolResult::error(vec![Content::text(
            "no bytes for that sentinel hash in the shared CAS (wrong hash, or it was never offloaded here)",
        )]),
        Err(e) => {
            CallToolResult::error(vec![Content::text(format!("shared CAS read failed: {e}"))])
        }
    }
}

#[tool_router]
impl Coffer {
    /// Run a shell command and hold its stdout/stderr server-side; returns a handle + summary.
    #[tool(
        description = "Run a shell command and hold its stdout/stderr SERVER-SIDE; returns a handle + a summary. \
        The output never enters your context — interrogate it with coffer_digest / coffer_query instead of reading it."
    )]
    async fn coffer_run(
        &self,
        Parameters(a): Parameters<RunArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        // Off by default: refuse before spawning unless explicitly enabled / allowlisted.
        if let Err(message) = run_policy_from_env().permits(&a.command) {
            return Ok(CallToolResult::error(vec![Content::text(message)]));
        }
        let limits = run_limits_from_env();
        match run_shell_command(&a.command, limits).await {
            Ok(capture) => {
                let h = self.put_bytes(&capture.bytes).as_str().to_string();
                let ingested = ingested_text(&h, &capture.bytes);
                let text = if capture.timed_out {
                    format!(
                        "command timed out after {} seconds; captured {} bytes before termination (partial output)\n{}",
                        limits.timeout_seconds,
                        capture.bytes.len(),
                        ingested
                    )
                } else if capture.output_truncated {
                    format!(
                        "command output exceeded COFFER_MCP_MAX_RUN_OUTPUT_MB-derived limit ({} bytes); captured first {} bytes and terminated the process (partial output)\n{}",
                        limits.max_output_bytes,
                        capture.bytes.len(),
                        ingested
                    )
                } else if capture.status.is_some_and(|status| status.success()) {
                    ingested
                } else if let Some(status) = capture.status {
                    format!("command exited {status}\n{ingested}")
                } else {
                    format!("command ended without an exit status\n{ingested}")
                };
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "spawn failed: {e}"
            ))])),
        }
    }

    /// Ingest a file from disk and hold it server-side; returns a handle + summary.
    #[tool(
        description = "Ingest a file from disk and hold it server-side; returns a handle + summary. \
        Optional view=\"structural_code\" returns an explicit compact code outline backed by the same retrievable original."
    )]
    async fn coffer_ingest(
        &self,
        Parameters(a): Parameters<IngestArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let view = match ingest_view(a.view.as_deref()) {
            Ok(view) => view,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        match std::fs::read(&a.path) {
            Ok(bytes) => {
                let h = self.put_bytes(&bytes).as_str().to_string();
                Ok(CallToolResult::success(vec![Content::text(
                    ingested_text_with_view(&h, &bytes, view, a.target_tokens, self),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "read {}: {e}",
                a.path
            ))])),
        }
    }

    /// Compute an EXACT deterministic aggregate over ALL of the held data.
    #[tool(
        description = "Compute an EXACT deterministic aggregate over ALL of the held data \
        (count/sum/mean/median/percentile/range/distinct/group-by/argmax/filter-aggregate). \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix. The result is COMPUTED, not estimated — trust it over your own count. \
        Returns nothing if no aggregate matches (it refuses rather than guess)."
    )]
    async fn coffer_digest(
        &self,
        Parameters(a): Parameters<DigestArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match self
            .dataset(&a.handle, &bytes)
            .and_then(|ds| ds.digest(&a.query))
        {
            Some(fact) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{fact}  (computed exactly over the whole dataset)"
            ))])),
            None => Ok(CallToolResult::error(vec![Content::text(
                "no deterministic aggregate matched this query over this data",
            )])),
        }
    }

    /// Shape-generic exact summary of a held JSON array of records (schema + per-field stats / count-by).
    #[tool(
        description = "Summarize a held JSON array of records GENERICALLY and exactly: row count, and per \
        field its present/distinct counts plus either numeric stats (min/max/mean/sum) or a \
        count-by-value breakdown for low-cardinality categoricals. No per-tool/per-format code — point it \
        at any record set (ideally a tool's structured --json/-o json output) for an RTK-style decision \
        summary that is exact and recoverable. Use coffer_aggregate for a specific number. handle may be a \
        full handle or a unique <<cof:HASH>> sentinel prefix over a JSON array."
    )]
    async fn coffer_describe(
        &self,
        Parameters(a): Parameters<DescribeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match self.dataset(&a.handle, &bytes).and_then(|ds| ds.describe()) {
            Some(card) => Ok(CallToolResult::success(vec![Content::text(card)])),
            None => Ok(CallToolResult::error(vec![Content::text(
                "coffer_describe needs a handle over a JSON array of records",
            )])),
        }
    }

    /// Keep only the rows where `field <op> value`; the rest are elided (still retrievable).
    #[tool(
        description = "Keep only the rows where field <op> value (op: eq|ne|gt|ge|lt|le) and return them; \
        non-matching rows are elided as placeholders. In SQLite/shared-CAS deployments, handle may be \
        a full handle or a unique <<cof:HASH>> sentinel prefix when that blob is valid JSON. \
        Byte-exact reversible — the originals stay retrievable."
    )]
    async fn coffer_query(
        &self,
        Parameters(a): Parameters<QueryArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let op = parse_op(&a.op);
        let value = parse_value_arg(&a.value);
        let doc = compress_json_where(&bytes, self, &a.field, op, value);
        Ok(CallToolResult::success(vec![Content::text(
            doc.render_for_model(),
        )]))
    }

    /// Filter a held JSON array by a CONJUNCTION of predicates and hold the matching rows as a NEW handle.
    #[tool(
        description = "Filter a held JSON array by a conjunction of predicates (where: a list of \
        {field, op, value}; op is eq|ne|gt|ge|lt|le; a row is kept only if it passes ALL of them) and \
        hold the matching rows as a NEW handle, returned with a fact card. The result is itself a \
        dataset: feed its handle back into coffer_select / coffer_digest / coffer_query / coffer_rows to \
        narrow further — all server-side, the rows never entering your context. Each kept row is byte-exact. \
        handle may be a full handle or a unique <<cof:HASH>> sentinel prefix over a JSON array."
    )]
    async fn coffer_select(
        &self,
        Parameters(a): Parameters<SelectArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let predicates = predicates_from_args(&a.predicates);
        let Some(subset) = query_subset(&bytes, &predicates) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "coffer_select needs a handle over a top-level JSON array",
            )]));
        };
        let handle = self.put_bytes(&subset).as_str().to_string();
        Ok(CallToolResult::success(vec![Content::text(ingested_text(
            &handle, &subset,
        ))]))
    }

    /// Exact typed aggregate over a held JSON array, returning the number AND its backing row indices.
    #[tool(
        description = "Exact aggregate over a held JSON array, typed and unambiguous: filter by a \
        conjunction of predicates (where: list of {field, op, value}; op eq|ne|gt|ge|lt|le; an ordering \
        op gt/ge/lt/le matches only when field and value are the same type, so compare a numeric field \
        with a number, not a quoted string) and \
        compute agg = count|sum|mean|min|max (field required for all but count). Computed over ALL \
        rows including offloaded ones — trust it over your own count. Returns the value AND the 0-based \
        indices of the backing records (provenance); feed those into coffer_pick(handle, indices) to \
        fetch exactly those rows and re-verify byte-for-byte. Refuses (no guess) when the aggregated \
        field is present-but-non-numeric, or the handle is not a JSON array. handle may be a full \
        handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_aggregate(
        &self,
        Parameters(a): Parameters<AggregateArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let Some(agg) = parse_agg(&a.agg, a.field.as_deref()) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "agg must be count|sum|mean|min|max, and sum/mean/min/max require a field",
            )]));
        };
        let predicates = predicates_from_args(&a.predicates);
        match self
            .dataset(&a.handle, &bytes)
            .and_then(|ds| ds.query_aggregate(&predicates, &agg))
        {
            Some(r) => {
                const SHOWN: usize = 64;
                let idx = if r.matched.len() <= SHOWN {
                    format!("{:?}", r.matched)
                } else {
                    let head: Vec<usize> = r.matched.iter().take(SHOWN).copied().collect();
                    format!("{head:?} … ({} total)", r.matched.len())
                };
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{}\nprovenance row indices: {idx}\nFetch them with coffer_pick(handle, indices) to re-verify.",
                    r.display
                ))]))
            }
            None => Ok(CallToolResult::error(vec![Content::text(
                "no exact aggregate: the handle is not a JSON array, or the aggregated field is present-but-non-numeric",
            )])),
        }
    }

    #[tool(
        description = "Exact NUMERIC BUCKETING histogram over a held JSON array: group rows into \
        fixed-width bands by floor(value / width) on a numeric field, then aggregate each band. \
        field = the numeric field to bucket on; width = the band width (> 0); agg = count|sum|mean|min|max \
        (value_field required for all but count), computed WITHIN each band. Bands are ordered by lower \
        bound and computed over ALL rows including offloaded ones — e.g. \"count per 100ms latency band\" \
        is bucket(field=latency_ms, width=100). Refuses (no guess) when width <= 0, the handle is not a \
        JSON array, or the aggregated field is present-but-non-numeric. handle may be a full handle or a \
        unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_bucket(
        &self,
        Parameters(a): Parameters<BucketArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let agg_name = a.agg.as_deref().unwrap_or("count");
        let Some(agg) = parse_agg(agg_name, a.value_field.as_deref()) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "agg must be count|sum|mean|min|max, and sum/mean/min/max require value_field",
            )]));
        };
        match self
            .dataset(&a.handle, &bytes)
            .and_then(|ds| ds.bucket_aggregate(&a.field, a.width, &agg))
        {
            Some(g) => Ok(CallToolResult::success(vec![Content::text(
                group_aggregate_text(&g),
            )])),
            None => Ok(CallToolResult::error(vec![Content::text(
                "no exact histogram: width must be > 0, the handle must be a JSON array, and the bucket/value field must be numeric",
            )])),
        }
    }

    #[tool(
        description = "Windowed LOG HISTOGRAM over a held text/log handle: count case-insensitive \
        substring matches of pattern per block of `window` lines, so you can see WHERE an event \
        clusters across a long log without reading it. Returns one row per block (0-based block index) \
        with the match count and the matching 1-based line numbers as provenance — e.g. \"ERROR lines \
        per 1000-line block\" is window(pattern=ERROR, window=1000). Refuses when window = 0 or pattern \
        is empty. handle may be a full handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_window(
        &self,
        Parameters(a): Parameters<WindowArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match coffer_core::count_matches_per_window(&bytes, &a.pattern, a.window) {
            Some(g) => Ok(CallToolResult::success(vec![Content::text(
                group_aggregate_text(&g),
            )])),
            None => Ok(CallToolResult::error(vec![Content::text(
                "no histogram: window must be >= 1 and pattern must be non-empty",
            )])),
        }
    }

    #[tool(
        description = "Exact correlation across TWO held JSON arrays without pulling either into your \
        context. SEMI-JOIN: aggregate the LEFT array over rows whose left_key has a matching right_key \
        in the RIGHT array. left/right = the two handles; left_key/right_key = the equi-join fields; \
        agg = count|sum|mean|min|max (field required for all but count) over the qualifying LEFT rows; \
        right_where = optional conjunctive filter on the RIGHT rows. Example: \"sum order.amount for \
        orders whose customer is gold-tier\" = join(left=orders, right=customers, left_key=customer_id, \
        right_key=id, agg=sum, field=amount, right_where=[{field:tier, op:eq, value:gold}]). Set group_by \
        to a RIGHT field for a PROJECT-JOIN group-by instead (\"revenue by customer.region\"); it refuses \
        rather than guessing when a join key maps to conflicting group values, and ignores right_where. A \
        duplicated join key never double-counts a left row. Each handle may be a full handle or a unique \
        <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_join(
        &self,
        Parameters(a): Parameters<JoinArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let (Some(left), Some(right)) = (self.get_handle(&a.left), self.get_handle(&a.right))
        else {
            return Ok(unknown_handle());
        };
        let agg_name = a.agg.as_deref().unwrap_or("count");
        let Some(agg) = parse_agg(agg_name, a.field.as_deref()) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "agg must be count|sum|mean|min|max, and sum/mean/min/max require a field",
            )]));
        };
        let joined = (self.dataset(&a.left, &left), self.dataset(&a.right, &right));
        let (Some(lds), Some(rds)) = joined else {
            return Ok(CallToolResult::error(vec![Content::text(
                "no exact join: both handles must be JSON arrays",
            )]));
        };
        if let Some(group_field) = a.group_by.as_deref().filter(|g| !g.is_empty()) {
            return match lds.join_group_aggregate(
                &rds,
                &a.left_key,
                &a.right_key,
                group_field,
                &agg,
            ) {
                Some(g) => Ok(CallToolResult::success(vec![Content::text(
                    group_aggregate_text(&g),
                )])),
                None => Ok(CallToolResult::error(vec![Content::text(
                    "no exact project-join: both handles must be JSON arrays, the aggregated field numeric, and no join key may map to conflicting group values",
                )])),
            };
        }
        let right_where = predicates_from_args(&a.right_where);
        match lds.join_aggregate(&rds, &a.left_key, &a.right_key, &right_where, &agg) {
            Some(r) => {
                const SHOWN: usize = 64;
                let idx = if r.matched.len() <= SHOWN {
                    format!("{:?}", r.matched)
                } else {
                    let head: Vec<usize> = r.matched.iter().take(SHOWN).copied().collect();
                    format!("{head:?} … ({} total)", r.matched.len())
                };
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{}\njoined LEFT row indices: {idx}\nFetch them with coffer_pick(left_handle, indices) to re-verify.",
                    r.display
                ))]))
            }
            None => Ok(CallToolResult::error(vec![Content::text(
                "no exact semi-join: both handles must be JSON arrays and the aggregated field numeric",
            )])),
        }
    }

    #[tool(
        description = "Check an agent's CLAIMED number against the held bytes: recompute the exact \
        aggregate (agg = count|sum|mean|min|max over the `where` predicate conjunction, field required \
        for all but count) and compare it to `expected`. Returns AGREE or DISAGREE with the exact value \
        and the backing row indices, so a number a model asserted can be confirmed or caught WITHOUT \
        trusting the model's arithmetic — computed over ALL rows including offloaded ones. Use it as a \
        lie-detector for tool-output: when an agent says 'there are 200 errors', check whether the held \
        data agrees. handle may be a full handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_check_claim(
        &self,
        Parameters(a): Parameters<CheckClaimArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let Some(agg) = parse_agg(&a.agg, a.field.as_deref()) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "agg must be count|sum|mean|min|max, and sum/mean/min/max require a field",
            )]));
        };
        let predicates = predicates_from_args(&a.predicates);
        match self
            .dataset(&a.handle, &bytes)
            .and_then(|ds| ds.query_aggregate(&predicates, &agg))
        {
            Some(r) => {
                let agree = (r.value - a.expected).abs() <= 1e-9 + 1e-9 * r.value.abs();
                const SHOWN: usize = 64;
                let idx = if r.matched.len() <= SHOWN {
                    format!("{:?}", r.matched)
                } else {
                    let head: Vec<usize> = r.matched.iter().take(SHOWN).copied().collect();
                    format!("{head:?} … ({} total)", r.matched.len())
                };
                let verdict = if agree { "AGREE" } else { "DISAGREE" };
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{verdict}\nclaimed: {}\nexact value: {}\nprovenance row indices: {idx}\nThe exact value is computed over all held bytes; coffer_pick(handle, indices) re-fetches the backing rows.",
                    a.expected, r.value,
                ))]))
            }
            None => Ok(CallToolResult::error(vec![Content::text(
                "cannot check: the handle is not a JSON array, or the aggregated field is present-but-non-numeric",
            )])),
        }
    }

    #[tool(
        description = "Issue a re-executable EXACTNESS RECEIPT for a typed aggregate over a held JSON \
        array (agg = count|sum|mean|min|max over the `where` predicate conjunction, field required for \
        all but count). Returns a small portable JSON proof binding the predicate + aggregate + value + \
        backing-row indices + a SHA-256 of the backing rows. Persist it and hand it to anyone: \
        coffer_verify_receipt re-derives the answer from the bytes later, in a fresh process, returning \
        VALID or a tamper signal — no model call. handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix."
    )]
    async fn coffer_receipt(
        &self,
        Parameters(a): Parameters<AggregateArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let Some(agg) = parse_agg(&a.agg, a.field.as_deref()) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "agg must be count|sum|mean|min|max, and sum/mean/min/max require a field",
            )]));
        };
        let predicates = predicates_from_args(&a.predicates);
        match coffer_core::issue_receipt(&bytes, &predicates, &agg) {
            Some(receipt) => match serde_json::to_string(&receipt) {
                Ok(wire) => Ok(CallToolResult::success(vec![Content::text(format!(
                    "{wire}\n\nStore this receipt. Re-verify it later against the data with coffer_verify_receipt(handle, receipt)."
                ))])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                    "could not serialize receipt: {e}"
                ))])),
            },
            None => Ok(CallToolResult::error(vec![Content::text(
                "no receipt: the handle is not a JSON array, or the aggregated field is present-but-non-numeric",
            )])),
        }
    }

    #[tool(
        description = "Verify a re-executable exactness receipt (from coffer_receipt) against a held \
        JSON array: re-run the receipt's query over the bytes and report VALID (the data reproduces the \
        attested answer byte-identically), VALUE_MISMATCH (the data yields a different number), \
        BACKING_TAMPERED (the value holds but the backing rows changed), REFUSED, or MALFORMED_RECEIPT. \
        No model call. `receipt` is the JSON string coffer_receipt returned. handle may be a full handle \
        or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_verify_receipt(
        &self,
        Parameters(a): Parameters<VerifyReceiptArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let receipt: coffer_core::Receipt = match serde_json::from_str(&a.receipt) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "receipt is not a valid coffer receipt JSON: {e}"
                ))]));
            }
        };
        let verdict = coffer_core::verify_receipt(&receipt, &bytes);
        let text = match &verdict {
            coffer_core::ReceiptVerdict::Valid => {
                "VALID — the held bytes reproduce the attested answer byte-identically.".to_string()
            }
            coffer_core::ReceiptVerdict::ValueMismatch { expected, actual } => format!(
                "VALUE_MISMATCH — the receipt attests {expected}, but the data now yields {actual}."
            ),
            coffer_core::ReceiptVerdict::BackingTampered => {
                "BACKING_TAMPERED — the value holds but the backing rows changed.".to_string()
            }
            coffer_core::ReceiptVerdict::Refused => {
                "REFUSED — the receipt's query no longer runs over this input.".to_string()
            }
            coffer_core::ReceiptVerdict::MalformedReceipt => {
                "MALFORMED_RECEIPT — the receipt carries an unknown op/agg and cannot be re-executed."
                    .to_string()
            }
        };
        // A mismatch is a real, useful answer, not a tool error — return it as success text.
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Pull the rows at an explicit set of indices (e.g. a digest's provenance) as a NEW handle.
    #[tool(
        description = "Pull the rows at an explicit set of indices from a held JSON array and hold them \
        as a NEW handle, returned with a fact card. Use it to AUDIT an aggregate: coffer_digest / a typed \
        query reports which record indices back its number, and coffer_pick(handle, indices) fetches \
        exactly those rows so you can re-verify byte-for-byte (e.g. coffer_digest the result to recount). \
        Rows keep the order given; an out-of-range index is refused. handle may be a full handle or a \
        unique <<cof:HASH>> sentinel prefix over a JSON array."
    )]
    async fn coffer_pick(
        &self,
        Parameters(a): Parameters<PickArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let Some(picked) = coffer_core::pick_rows(&bytes, &a.indices) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "coffer_pick needs a JSON-array handle and in-range indices",
            )]));
        };
        let handle = self.put_bytes(&picked).as_str().to_string();
        Ok(CallToolResult::success(vec![Content::text(ingested_text(
            &handle, &picked,
        ))]))
    }

    /// Return a small row window from a held JSON array.
    #[tool(
        description = "Return rows start..start+limit from a held JSON array. Defaults to start=0, limit=20. \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix when that blob is a JSON array. Use before coffer_retrieve when you only \
        need local examples or a page of rows."
    )]
    async fn coffer_rows(
        &self,
        Parameters(a): Parameters<RowsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_json_rows(&bytes, a.start, a.limit, max_rows_from_env()) {
            Ok(rows) => Ok(CallToolResult::success(vec![Content::text(rows)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Return one value from held JSON by a small path syntax: `$.field[0].field`.
    #[tool(
        description = "Return one value from held JSON by path. Supported syntax: $, .field, and [index], e.g. $.items[0].name. \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix when that blob is valid JSON."
    )]
    async fn coffer_json(
        &self,
        Parameters(a): Parameters<JsonPathArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_json_path(&bytes, &a.path) {
            Ok(value) => Ok(CallToolResult::success(vec![Content::text(value)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Return line-numbered text/log lines from a held payload.
    #[tool(
        description = "Return line-numbered text/log lines from a held payload. \
        Uses 1-based inclusive start_line/end_line, or head/tail for first/last N lines. \
        Defaults to the first 80 lines. In SQLite/shared-CAS deployments, handle may be a full \
        handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_lines(
        &self,
        Parameters(a): Parameters<LinesArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_text_lines(&bytes, a.start_line, a.end_line, a.head, a.tail) {
            Ok(lines) => Ok(CallToolResult::success(vec![Content::text(lines)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Search held text/log output and return matching line numbers.
    #[tool(
        description = "Search held text/log output and return matching line numbers. \
        Case-insensitive substring search; limit defaults to 20. In SQLite/shared-CAS deployments, \
        handle may be a full handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_search(
        &self,
        Parameters(a): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_text_search(&bytes, &a.pattern, a.limit) {
            Ok(lines) => Ok(CallToolResult::success(vec![Content::text(lines)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Return a byte-exact window from the held original payload.
    #[tool(
        description = "Return a byte-exact window from the original bytes held under the handle. \
        By default returns only a bounded head window. Optional start/max_bytes returns that byte window. \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix. Set full=true only for small payloads you truly need raw; full retrieval is capped by COFFER_MCP_MAX_RETRIEVE_BYTES."
    )]
    async fn coffer_retrieve(
        &self,
        Parameters(a): Parameters<RetrieveArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.get_handle(&a.handle) {
            Some(bytes) => match render_retrieved_bytes(
                &bytes,
                a.start,
                a.max_bytes,
                a.full.unwrap_or(false),
                retrieve_limits_from_env(),
            ) {
                Ok(text) => Ok(CallToolResult::success(vec![Content::text(text)])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
            },
            None => Ok(unknown_handle()),
        }
    }

    /// Recover a byte-exact window from bytes elided behind a `<<cof:HASH …>>` compression sentinel.
    #[tool(
        description = "Recover a byte-exact window from bytes elided behind a <<cof:HASH …>> sentinel \
        (e.g. in a coffer-compressed tool output that a proxy shrank). Pass the HASH shown in the \
        sentinel; returns a bounded byte window from the shared CAS by default. Optional start/max_bytes chooses the window. Set full=true only for small payloads; full retrieval is capped by COFFER_MCP_MAX_RETRIEVE_BYTES."
    )]
    async fn coffer_unfold(
        &self,
        Parameters(a): Parameters<UnfoldArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Ok(path) = std::env::var("COFFER_CAS_DB") else {
            return Ok(CallToolResult::error(vec![Content::text(
                "no shared CAS configured (set COFFER_CAS_DB to the proxy's database path)",
            )]));
        };
        Ok(unfold_shared_cas_result(
            path,
            &a,
            retrieve_limits_from_env(),
        ))
    }

    /// Report handle-store backend and durability metrics.
    #[tool(
        description = "Report coffer handle-store backend, resident bytes, handle count, and SQLite durability metrics."
    )]
    async fn coffer_status(&self) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![Content::text(
            self.status_text(),
        )]))
    }
}

#[tool_handler(
    name = "coffer",
    instructions = "coffer holds large tool-output SERVER-SIDE so it never fills your context. \
        Direct it at data instead of reading it: coffer_run / coffer_ingest return an opaque handle \
        plus a fact card; then coffer_digest gives EXACT aggregates over ALL of it (count/sum/mean/median/\
        percentile/group-by/argmax — trust these over your own count), coffer_describe gives a generic \
        exact summary of any record set (schema + per-field stats / count-by), coffer_aggregate gives a typed \
        count|sum|mean|min|max over a predicate conjunction and returns the answer WITH its backing row \
        indices, coffer_query keeps only the rows \
        matching field <op> value, coffer_select filters by a conjunction of predicates and returns the \
        matches as a NEW handle you can narrow again server-side, coffer_pick pulls rows at explicit indices \
        (a digest's provenance) so you can audit an aggregate, coffer_search/coffer_lines drill into logs and text by line number, \
        coffer_json returns one JSON value, coffer_rows returns a small JSON row window, and coffer_retrieve \
        returns bounded byte windows. In SQLite/shared-CAS deployments those handle-taking tools can also \
        accept a unique hash prefix from a <<cof:...>> proxy sentinel; use coffer_unfold for explicit \
        sentinel byte windows. Whole-payload retrieval requires full=true and is capped. Use coffer_status for backend/durability diagnostics."
)]
impl ServerHandler for Coffer {}

#[derive(Deserialize, JsonSchema)]
struct RunArgs {
    /// Shell command whose stdout/stderr to capture and hold server-side.
    command: String,
}
#[derive(Deserialize, JsonSchema)]
struct IngestArgs {
    /// Path to a file to ingest.
    path: String,
    /// Optional compact view to return with the handle. Supported: structural_code.
    view: Option<String>,
    /// Optional token target for the compact view. Defaults to 1024 for structural_code.
    target_tokens: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct DigestArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// A natural-language aggregate, e.g. "how many commits changed more than 50 files".
    query: String,
}
#[derive(Deserialize, JsonSchema)]
struct QueryArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// The JSON field to filter on.
    field: String,
    /// Comparison operator: eq | ne | gt | ge | lt | le.
    op: String,
    /// The value to compare against (a number or string).
    value: String,
}
#[derive(Deserialize, JsonSchema)]
struct SelectArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Conjunctive predicates; a row is kept only if it passes ALL of them. An empty list keeps every row.
    #[serde(rename = "where", default)]
    predicates: Vec<PredicateArg>,
}
#[derive(Deserialize, JsonSchema)]
struct DescribeArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
}
#[derive(Deserialize, JsonSchema)]
struct AggregateArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Conjunctive filter predicates; a row is counted only if it passes ALL of them. Empty = all rows.
    #[serde(rename = "where", default)]
    predicates: Vec<PredicateArg>,
    /// Aggregate to compute: count | sum | mean | min | max.
    agg: String,
    /// Field to aggregate over. Required for sum/mean/min/max; ignored for count.
    field: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
struct PredicateArg {
    /// The JSON field to filter on.
    field: String,
    /// Comparison operator: eq | ne | gt | ge | lt | le.
    op: String,
    /// The value to compare against (a number or string).
    value: String,
}
#[derive(Deserialize, JsonSchema)]
struct PickArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Zero-based record indices to pull (e.g. the provenance indices reported by a typed query). Order is preserved.
    indices: Vec<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct BucketArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// The numeric field to bucket on; rows are grouped by floor(value / width).
    field: String,
    /// The band width; must be > 0.
    width: f64,
    /// Aggregate to compute within each band: count | sum | mean | min | max. Defaults to count.
    agg: Option<String>,
    /// Field to aggregate within each band. Required for sum/mean/min/max; ignored for count.
    value_field: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
struct WindowArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// The pattern to match (case-insensitive substring) on each line.
    pattern: String,
    /// Block size in lines; match counts are reported per block of this many lines. Must be >= 1.
    window: usize,
}
#[derive(Deserialize, JsonSchema)]
struct JoinArgs {
    /// The LEFT array handle — the rows being aggregated.
    left: String,
    /// The RIGHT array handle — the lookup/filter side.
    right: String,
    /// The equi-join field on the LEFT rows.
    left_key: String,
    /// The equi-join field on the RIGHT rows.
    right_key: String,
    /// Aggregate over the qualifying LEFT rows: count | sum | mean | min | max. Defaults to count.
    agg: Option<String>,
    /// Field to aggregate over the LEFT rows. Required for sum/mean/min/max; ignored for count.
    field: Option<String>,
    /// Optional conjunctive filter on the RIGHT rows (ignored when group_by is set).
    #[serde(default)]
    right_where: Vec<PredicateArg>,
    /// Set to a RIGHT field for a project-join group-by ("agg BY right.<field>") instead of a scalar.
    group_by: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
struct CheckClaimArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Conjunctive filter predicates; a row is counted only if it passes ALL of them. Empty = all rows.
    #[serde(rename = "where", default)]
    predicates: Vec<PredicateArg>,
    /// Aggregate to recompute and compare: count | sum | mean | min | max.
    agg: String,
    /// Field to aggregate over. Required for sum/mean/min/max; ignored for count.
    field: Option<String>,
    /// The number the agent claimed; AGREE if it matches the exact recomputed value, else DISAGREE.
    expected: f64,
}
#[derive(Deserialize, JsonSchema)]
struct VerifyReceiptArgs {
    /// The handle holding the data to re-verify against, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// The receipt JSON string returned by coffer_receipt.
    receipt: String,
}
#[derive(Deserialize, JsonSchema)]
struct RowsArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Optional zero-based row offset.
    start: Option<usize>,
    /// Optional maximum number of rows to return. Defaults to 20.
    limit: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct JsonPathArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// JSON path using `$`, `.field`, and `[index]`, e.g. `$.items[0].name`.
    path: String,
}
#[derive(Deserialize, JsonSchema)]
struct LinesArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Optional 1-based first line to return.
    start_line: Option<usize>,
    /// Optional 1-based inclusive last line to return.
    end_line: Option<usize>,
    /// Optional number of first lines to return.
    head: Option<usize>,
    /// Optional number of last lines to return.
    tail: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct SearchArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Case-insensitive substring to search for.
    pattern: String,
    /// Optional maximum matching lines to return. Defaults to 20.
    limit: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct RetrieveArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Optional byte offset to start returning from.
    start: Option<usize>,
    /// Optional maximum number of bytes to return.
    max_bytes: Option<usize>,
    /// Return the entire payload only when it is under the configured hard cap.
    full: Option<bool>,
}
#[derive(Deserialize, JsonSchema)]
struct UnfoldArgs {
    /// The hash shown inside a `<<cof:HASH …>>` sentinel (a hex prefix).
    hash: String,
    /// Optional byte offset to start returning from.
    start: Option<usize>,
    /// Optional maximum number of bytes to return.
    max_bytes: Option<usize>,
    /// Return the entire payload only when it is under the configured hard cap.
    full: Option<bool>,
}

/// Install a stderr-only, env-filtered tracing subscriber so coffer-cas durability and
/// corruption warnings reach operators. **Stderr only**: stdout is the MCP JSON-RPC channel and any
/// stray byte on it corrupts the protocol. Filter precedence: `RUST_LOG`, then `COFFER_LOG`, then
/// `default_directives`. Fail-open: a bad filter falls back to the default and `try_init` is a no-op
/// if a subscriber is already installed (e.g. under a test harness).
fn init_tracing(default_directives: &str) {
    let directives = std::env::var("RUST_LOG")
        .or_else(|_| std::env::var("COFFER_LOG"))
        .unwrap_or_else(|_| default_directives.to_string());
    let filter = tracing_subscriber::EnvFilter::try_new(&directives)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_directives));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Default to warn: keep the JSON-RPC peer's stderr quiet, but surface coffer-cas durability /
    // corruption warnings (e.g. an offloaded original that failed to persist) at failure time.
    init_tracing("warn");
    let service = Coffer::new()?.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AggregateArgs, BucketArgs, CheckClaimArgs, Coffer, DEFAULT_MAX_ROWS, DescribeArgs,
        IngestView, JoinArgs, PickArgs, PredicateArg, QueryArgs, RetrieveLimits, RunLimits,
        SelectArgs, UnfoldArgs, VerifyReceiptArgs, WindowArgs, byte_window, fact_card, ingest_view,
        ingested_text_with_view, positive_usize_from_value, render_json_path, render_json_rows,
        render_retrieved_bytes, render_text_lines, render_text_search, retrieve_limits_from_values,
        run_limits_from_values, run_policy_from_values, run_shell_command,
        unfold_shared_cas_result,
    };
    use coffer_cas::{Cas, ContentHash, MemoryCas, SqliteCas};
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::CallToolResult;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("coffer-mcp-{}-{tag}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn db(&self) -> std::path::PathBuf {
            self.0.join("cas.db")
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tool_text(result: &CallToolResult) -> &str {
        result.content[0].as_text().unwrap().text.as_str()
    }

    #[test]
    fn ingest_view_rejects_unknown_values() {
        let err = ingest_view(Some("whole_repo_map")).unwrap_err();
        assert!(err.contains("supported values: structural_code"), "{err}");
    }

    #[test]
    fn structural_code_ingest_view_keeps_outline_and_retrieval_sentinel() {
        let input = br#"
use std::collections::HashMap;

pub struct Widget {
    value: usize,
}

impl Widget {
    pub fn compute(&self, items: &[usize]) -> usize {
        let mut total = self.value;
        for item in items {
            total += item;
        }
        total
    }
}
"#;
        let handle = ContentHash::of(input).as_str().to_string();
        let cas = MemoryCas::new();
        let text =
            ingested_text_with_view(&handle, input, IngestView::StructuralCode, Some(120), &cas);
        assert!(text.contains(&format!("handle: {handle}")), "{text}");
        assert!(text.contains(&format!("<<cof:{}", &handle[..12])), "{text}");
        assert!(text.contains("view: structural_code"), "{text}");
        assert!(text.contains("pub struct Widget"), "{text}");
        assert!(text.contains("pub fn compute"), "{text}");
        assert!(!text.contains("total += item"), "{text}");
    }

    #[tokio::test]
    async fn query_sentinel_is_retrievable_from_sqlite_store() {
        let dir = TempDir::new("query-sentinel");
        let server = Coffer::with_sqlite(dir.db()).unwrap();
        let rows = (0..30)
            .map(|i| {
                serde_json::json!({
                    "id": i,
                    "status": if i == 17 { "open" } else { "closed" },
                    "payload": format!("row-{i:02}-{}", "x".repeat(200)),
                })
            })
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        let result = server
            .coffer_query(Parameters(QueryArgs {
                handle,
                field: "id".to_string(),
                op: "eq".to_string(),
                value: "17".to_string(),
            }))
            .await
            .unwrap();
        let text = tool_text(&result);
        let prefix = text
            .split_once("<<cof:")
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .expect("query should elide non-matching rows");

        let result = unfold_shared_cas_result(
            dir.db(),
            &UnfoldArgs {
                hash: prefix.to_string(),
                start: Some(0),
                max_bytes: Some(80),
                full: None,
            },
            RetrieveLimits {
                default_bytes: 80,
                max_bytes: 256,
            },
        );

        assert_eq!(result.is_error, Some(false), "{}", tool_text(&result));
        assert!(
            tool_text(&result).contains("\"id\":0"),
            "{}",
            tool_text(&result)
        );
    }

    fn handle_of(card: &str) -> String {
        card.lines()
            .next()
            .unwrap()
            .strip_prefix("handle: ")
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn coffer_select_chains_handles_server_side() {
        let server = Coffer::in_memory();
        let rows = (0..20)
            .map(|i| {
                serde_json::json!({
                    "a": i,
                    "status": if i % 2 == 0 { "error" } else { "ok" },
                })
            })
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        // select a > 10 → a NEW handle (the derived subset lives in the shared CAS).
        let r1 = server
            .coffer_select(Parameters(SelectArgs {
                handle,
                predicates: vec![PredicateArg {
                    field: "a".into(),
                    op: "gt".into(),
                    value: "10".into(),
                }],
            }))
            .await
            .unwrap();
        assert_eq!(r1.is_error, Some(false), "{}", tool_text(&r1));
        let h1 = handle_of(tool_text(&r1));

        // chain: select status == "error" on the DERIVED handle, server-side.
        let r2 = server
            .coffer_select(Parameters(SelectArgs {
                handle: h1,
                predicates: vec![PredicateArg {
                    field: "status".into(),
                    op: "eq".into(),
                    value: "error".into(),
                }],
            }))
            .await
            .unwrap();
        assert_eq!(r2.is_error, Some(false), "{}", tool_text(&r2));
        let h2 = handle_of(tool_text(&r2));

        // the chained handle equals applying both predicates at once (composition holds across handles).
        let got = server.get_handle(&h2).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&got).unwrap();
        let want: Vec<serde_json::Value> = rows
            .iter()
            .filter(|r| r["a"].as_i64().unwrap() > 10 && r["status"].as_str() == Some("error"))
            .cloned()
            .collect();
        assert_eq!(parsed, want);
        // a > 10 and even: 12, 14, 16, 18 → 4 rows.
        assert_eq!(parsed.len(), 4);
    }

    #[tokio::test]
    async fn coffer_describe_summarizes_a_records_handle() {
        let server = Coffer::in_memory();
        let rows = (0..20)
            .map(|i| serde_json::json!({ "v": i, "status": if i % 4 == 0 { "error" } else { "ok" } }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        let r = server
            .coffer_describe(Parameters(DescribeArgs { handle }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        let t = tool_text(&r);
        assert!(t.contains("20 records"), "{t}");
        assert!(t.contains("sum=190"), "{t}"); // 0+1+..+19
        assert!(
            t.contains(r#""error":5"#) && t.contains(r#""ok":15"#),
            "{t}"
        ); // exact count-by

        // non-array handle is refused.
        let h2 = server.put_bytes(b"not json").as_str().to_string();
        let bad = server
            .coffer_describe(Parameters(DescribeArgs { handle: h2 }))
            .await
            .unwrap();
        assert_eq!(bad.is_error, Some(true), "{}", tool_text(&bad));
    }

    #[tokio::test]
    async fn coffer_bucket_and_window_histograms_wire_through() {
        let server = Coffer::in_memory();
        // latencies 0,50,100,…,450 → width-100 bands: [0,2) [100,2) [200,2) [300,2) [400,2).
        let rows = (0..10)
            .map(|i| serde_json::json!({ "latency_ms": i * 50 }))
            .collect::<Vec<_>>();
        let handle = server
            .put_bytes(&serde_json::to_vec(&rows).unwrap())
            .as_str()
            .to_string();
        let r = server
            .coffer_bucket(Parameters(BucketArgs {
                handle,
                field: "latency_ms".into(),
                width: 100.0,
                agg: None,
                value_field: None,
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        // Five bands, each with two rows.
        assert_eq!(tool_text(&r).matches('→').count(), 5, "{}", tool_text(&r));

        // width <= 0 refuses rather than guessing.
        let h = server
            .put_bytes(&serde_json::to_vec(&rows).unwrap())
            .as_str()
            .to_string();
        let bad = server
            .coffer_bucket(Parameters(BucketArgs {
                handle: h,
                field: "latency_ms".into(),
                width: 0.0,
                agg: None,
                value_field: None,
            }))
            .await
            .unwrap();
        assert_eq!(bad.is_error, Some(true), "{}", tool_text(&bad));

        // Windowed log histogram: 2 ERROR lines, one per 2-line block.
        let log = b"INFO ok\nERROR boom\nINFO ok\nERROR again\nINFO fine\n";
        let h2 = server.put_bytes(log).as_str().to_string();
        let w = server
            .coffer_window(Parameters(WindowArgs {
                handle: h2,
                pattern: "error".into(),
                window: 2,
            }))
            .await
            .unwrap();
        assert_eq!(w.is_error, Some(false), "{}", tool_text(&w));
        assert_eq!(tool_text(&w).matches('→').count(), 2, "{}", tool_text(&w));
    }

    #[tokio::test]
    async fn coffer_join_semi_join_and_project_group() {
        let server = Coffer::in_memory();
        let orders = serde_json::to_vec(&serde_json::json!([
            { "customer_id": 1, "amount": 100 },
            { "customer_id": 2, "amount": 50 },
            { "customer_id": 1, "amount": 25 },
            { "customer_id": 3, "amount": 999 }
        ]))
        .unwrap();
        let customers = serde_json::to_vec(&serde_json::json!([
            { "id": 1, "tier": "gold", "region": "us" },
            { "id": 2, "tier": "silver", "region": "us" },
            { "id": 3, "tier": "gold", "region": "eu" }
        ]))
        .unwrap();
        let left = server.put_bytes(&orders).as_str().to_string();
        let right = server.put_bytes(&customers).as_str().to_string();

        // sum order.amount for orders whose customer is gold-tier → customers 1,3 → 100+25+999 = 1124.
        let r = server
            .coffer_join(Parameters(JoinArgs {
                left: left.clone(),
                right: right.clone(),
                left_key: "customer_id".into(),
                right_key: "id".into(),
                agg: Some("sum".into()),
                field: Some("amount".into()),
                right_where: vec![PredicateArg {
                    field: "tier".into(),
                    op: "eq".into(),
                    value: "gold".into(),
                }],
                group_by: None,
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        assert!(tool_text(&r).contains("1124"), "{}", tool_text(&r));
        assert!(
            tool_text(&r).contains("joined LEFT row indices"),
            "{}",
            tool_text(&r)
        );

        // project-join group-by: revenue BY customer.region → us 100+50+25=175, eu 999.
        let g = server
            .coffer_join(Parameters(JoinArgs {
                left,
                right,
                left_key: "customer_id".into(),
                right_key: "id".into(),
                agg: Some("sum".into()),
                field: Some("amount".into()),
                right_where: vec![],
                group_by: Some("region".into()),
            }))
            .await
            .unwrap();
        assert_eq!(g.is_error, Some(false), "{}", tool_text(&g));
        let gt = tool_text(&g);
        assert!(gt.contains("175") && gt.contains("999"), "{gt}");
    }

    #[tokio::test]
    async fn coffer_check_claim_agrees_and_disagrees() {
        let server = Coffer::in_memory();
        let rows = serde_json::json!([
            { "status": "ok" }, { "status": "error" }, { "status": "error" }
        ]);
        let handle = server
            .put_bytes(&serde_json::to_vec(&rows).unwrap())
            .as_str()
            .to_string();

        // The true count of error rows is 2.
        let agree = server
            .coffer_check_claim(Parameters(CheckClaimArgs {
                handle: handle.clone(),
                predicates: vec![PredicateArg {
                    field: "status".into(),
                    op: "eq".into(),
                    value: "error".into(),
                }],
                agg: "count".into(),
                field: None,
                expected: 2.0,
            }))
            .await
            .unwrap();
        assert_eq!(agree.is_error, Some(false), "{}", tool_text(&agree));
        assert!(tool_text(&agree).contains("AGREE"), "{}", tool_text(&agree));

        let disagree = server
            .coffer_check_claim(Parameters(CheckClaimArgs {
                handle,
                predicates: vec![PredicateArg {
                    field: "status".into(),
                    op: "eq".into(),
                    value: "error".into(),
                }],
                agg: "count".into(),
                field: None,
                expected: 9.0,
            }))
            .await
            .unwrap();
        let dt = tool_text(&disagree);
        assert!(
            dt.contains("DISAGREE") && dt.contains("exact value: 2"),
            "{dt}"
        );
    }

    #[tokio::test]
    async fn coffer_receipt_round_trips_and_detects_tamper() {
        let server = Coffer::in_memory();
        let rows = serde_json::json!([
            { "status": "ok", "cost": 10 },
            { "status": "error", "cost": 40 },
            { "status": "error", "cost": 60 }
        ]);
        let handle = server
            .put_bytes(&serde_json::to_vec(&rows).unwrap())
            .as_str()
            .to_string();

        // Issue a receipt for sum(cost) where status == error (= 100).
        let issued = server
            .coffer_receipt(Parameters(AggregateArgs {
                handle: handle.clone(),
                predicates: vec![PredicateArg {
                    field: "status".into(),
                    op: "eq".into(),
                    value: "error".into(),
                }],
                agg: "sum".into(),
                field: Some("cost".into()),
            }))
            .await
            .unwrap();
        assert_eq!(issued.is_error, Some(false), "{}", tool_text(&issued));
        // The receipt JSON is the first line of the response.
        let receipt_json = tool_text(&issued).lines().next().unwrap().to_string();

        // Verify against the original data -> VALID.
        let ok = server
            .coffer_verify_receipt(Parameters(VerifyReceiptArgs {
                handle,
                receipt: receipt_json.clone(),
            }))
            .await
            .unwrap();
        assert!(tool_text(&ok).contains("VALID"), "{}", tool_text(&ok));

        // Verify the SAME receipt against tampered data held under a new handle -> VALUE_MISMATCH.
        let tampered = serde_json::json!([
            { "status": "ok", "cost": 10 },
            { "status": "error", "cost": 999 },
            { "status": "error", "cost": 60 }
        ]);
        let bad_handle = server
            .put_bytes(&serde_json::to_vec(&tampered).unwrap())
            .as_str()
            .to_string();
        let bad = server
            .coffer_verify_receipt(Parameters(VerifyReceiptArgs {
                handle: bad_handle,
                receipt: receipt_json,
            }))
            .await
            .unwrap();
        assert!(
            tool_text(&bad).contains("VALUE_MISMATCH"),
            "{}",
            tool_text(&bad)
        );
    }

    #[tokio::test]
    async fn coffer_aggregate_reports_value_and_provenance() {
        let server = Coffer::in_memory();
        let rows = (0..10)
            .map(|i| serde_json::json!({ "a": i, "b": i * 2 }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        // sum b where a >= 7  -> rows 7,8,9 -> b = 14+16+18 = 48
        let r = server
            .coffer_aggregate(Parameters(AggregateArgs {
                handle,
                predicates: vec![PredicateArg {
                    field: "a".into(),
                    op: "ge".into(),
                    value: "7".into(),
                }],
                agg: "sum".into(),
                field: Some("b".into()),
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        let text = tool_text(&r);
        assert!(text.contains("= 48"), "{text}");
        // provenance indices (7,8,9) are reported so they can feed coffer_pick.
        assert!(text.contains("provenance row indices"), "{text}");
        assert!(
            text.contains('7') && text.contains('8') && text.contains('9'),
            "{text}"
        );

        // a non-numeric aggregated field refuses rather than guessing.
        let strs = serde_json::to_vec(&serde_json::json!([{ "a": 1, "b": "x" }])).unwrap();
        let h2 = server.put_bytes(&strs).as_str().to_string();
        let bad = server
            .coffer_aggregate(Parameters(AggregateArgs {
                handle: h2,
                predicates: vec![],
                agg: "sum".into(),
                field: Some("b".into()),
            }))
            .await
            .unwrap();
        assert_eq!(bad.is_error, Some(true), "{}", tool_text(&bad));
    }

    #[tokio::test]
    async fn coffer_aggregate_wiring_edges_and_pick_roundtrip() {
        let server = Coffer::in_memory();
        let rows = (0..6)
            .map(|i| serde_json::json!({ "a": i }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();
        let agg = |op: &str, field: Option<&str>| AggregateArgs {
            handle: handle.clone(),
            predicates: vec![PredicateArg {
                field: "a".into(),
                op: "ge".into(),
                value: "3".into(),
            }],
            agg: op.to_string(),
            field: field.map(str::to_string),
        };

        // sum/mean/min/max WITHOUT a field error (never silently count); unknown agg errors.
        for op in ["sum", "mean", "min", "max", "bogus"] {
            let r = server
                .coffer_aggregate(Parameters(agg(op, None)))
                .await
                .unwrap();
            assert_eq!(r.is_error, Some(true), "{op} without field must error");
        }
        // count needs no field.
        let c = server
            .coffer_aggregate(Parameters(agg("count", None)))
            .await
            .unwrap();
        assert_eq!(c.is_error, Some(false), "{}", tool_text(&c));

        // "avg" aliases mean: mean of a in {3,4,5} = 4.
        let m = server
            .coffer_aggregate(Parameters(agg("avg", Some("a"))))
            .await
            .unwrap();
        assert!(tool_text(&m).contains("= 4"), "{}", tool_text(&m));

        // provenance round-trip: the reported indices, fed to coffer_pick, return exactly those rows.
        let prov = server
            .coffer_aggregate(Parameters(agg("count", None)))
            .await
            .unwrap();
        let text = tool_text(&prov);
        assert!(text.contains("provenance row indices: [3, 4, 5]"), "{text}");
        let picked = server
            .coffer_pick(Parameters(PickArgs {
                handle: handle.clone(),
                indices: vec![3, 4, 5],
            }))
            .await
            .unwrap();
        let bytes = server.get_handle(&handle_of(tool_text(&picked))).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed,
            vec![rows[3].clone(), rows[4].clone(), rows[5].clone()]
        );
    }

    #[tokio::test]
    async fn coffer_pick_pulls_provenance_rows() {
        let server = Coffer::in_memory();
        let rows = (0..6)
            .map(|i| serde_json::json!({ "id": i, "v": i * 10 }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        // pull records 1 and 4 (e.g. a digest's provenance), as a NEW handle.
        let r = server
            .coffer_pick(Parameters(PickArgs {
                handle,
                indices: vec![1, 4],
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        let picked = server.get_handle(&handle_of(tool_text(&r))).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&picked).unwrap();
        assert_eq!(parsed, vec![rows[1].clone(), rows[4].clone()]);
    }

    #[test]
    fn fact_card_includes_fields_absent_from_first_row() {
        let bytes = br#"[
            {"id": 1, "status": "seed"},
            {"id": 2, "latency_ms": 120, "status": "ok"},
            {"id": 3, "latency_ms": 240, "error_code": "E42"}
        ]"#;

        let card = fact_card(bytes).expect("JSON rows should produce field stats");

        assert!(
            card.contains("id: numeric present=3/3 min=1 max=3 mean=2.0000"),
            "{card}"
        );
        assert!(card.contains("status: present=2/3 2 distinct"), "{card}");
        assert!(
            card.contains("latency_ms: numeric present=2/3 min=120 max=240 mean=180.0000"),
            "{card}"
        );
        assert!(
            card.contains("error_code: present=1/3 1 distinct"),
            "{card}"
        );
    }

    #[test]
    fn byte_window_defaults_to_full_payload() {
        let window = byte_window(b"abcdef", None, None);
        assert_eq!(window.start, 0);
        assert_eq!(window.end, 6);
        assert_eq!(window.total, 6);
        assert_eq!(window.bytes, b"abcdef");
    }

    #[test]
    fn byte_window_applies_start_and_max_bytes() {
        let window = byte_window(b"abcdef", Some(2), Some(3));
        assert_eq!(window.start, 2);
        assert_eq!(window.end, 5);
        assert_eq!(window.total, 6);
        assert_eq!(window.bytes, b"cde");
    }

    #[test]
    fn byte_window_clamps_out_of_range_start() {
        let window = byte_window(b"abc", Some(99), Some(10));
        assert_eq!(window.start, 3);
        assert_eq!(window.end, 3);
        assert_eq!(window.total, 3);
        assert_eq!(window.bytes, b"");
    }

    #[test]
    fn render_full_payload_preserves_legacy_no_header_shape() {
        let limits = RetrieveLimits {
            default_bytes: 64,
            max_bytes: 64,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", None, None, false, limits).unwrap(),
            "abcdef"
        );
    }

    #[test]
    fn render_window_includes_offsets_and_omission_counts() {
        let limits = RetrieveLimits {
            default_bytes: 64,
            max_bytes: 64,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", Some(1), Some(2), false, limits).unwrap(),
            "bytes 1..3 of 6 (1 before, 3 after)\nbc"
        );
    }

    #[test]
    fn render_default_retrieve_is_bounded_window() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 10,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", None, None, false, limits).unwrap(),
            "bytes 0..3 of 6 (0 before, 3 after)\nabc"
        );
    }

    #[test]
    fn render_full_retrieve_requires_payload_under_hard_cap() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 6,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", None, None, true, limits).unwrap(),
            "abcdef"
        );
        let err = render_retrieved_bytes(b"abcdefg", None, None, true, limits).unwrap_err();
        assert!(err.contains("COFFER_MCP_MAX_RETRIEVE_BYTES=6"), "{err}");
    }

    #[test]
    fn render_rejects_window_over_hard_cap() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 4,
        };
        let err = render_retrieved_bytes(b"abcdef", Some(0), Some(5), false, limits).unwrap_err();
        assert_eq!(
            err,
            "requested max_bytes=5 exceeds COFFER_MCP_MAX_RETRIEVE_BYTES=4"
        );
    }

    #[test]
    fn render_full_rejects_window_arguments() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 10,
        };
        let err = render_retrieved_bytes(b"abcdef", Some(1), None, true, limits).unwrap_err();
        assert_eq!(err, "full=true cannot be combined with start or max_bytes");
    }

    #[test]
    fn unfold_shared_cas_returns_bounded_window_from_sentinel_hash() {
        let dir = TempDir::new("unfold-window");
        let cas = SqliteCas::open(dir.db()).unwrap();
        let hash = cas.put(b"0123456789");
        cas.flush();

        let result = unfold_shared_cas_result(
            dir.db(),
            &UnfoldArgs {
                hash: hash.short().to_string(),
                start: Some(3),
                max_bytes: Some(4),
                full: None,
            },
            RetrieveLimits {
                default_bytes: 3,
                max_bytes: 8,
            },
        );

        assert_eq!(result.is_error, Some(false));
        assert_eq!(
            tool_text(&result),
            "bytes 3..7 of 10 (3 before, 3 after)\n3456"
        );
    }

    #[test]
    fn unfold_shared_cas_enforces_full_retrieve_cap() {
        let dir = TempDir::new("unfold-full-cap");
        let cas = SqliteCas::open(dir.db()).unwrap();
        let hash = cas.put(b"0123456789");
        cas.flush();

        let result = unfold_shared_cas_result(
            dir.db(),
            &UnfoldArgs {
                hash: hash.short().to_string(),
                start: None,
                max_bytes: None,
                full: Some(true),
            },
            RetrieveLimits {
                default_bytes: 3,
                max_bytes: 8,
            },
        );

        assert_eq!(result.is_error, Some(true));
        assert!(
            tool_text(&result).contains("COFFER_MCP_MAX_RETRIEVE_BYTES=8"),
            "{}",
            tool_text(&result)
        );
    }

    #[test]
    fn sqlite_store_accepts_unique_sentinel_prefix_as_handle() {
        let dir = TempDir::new("sqlite-prefix-handle");
        let hash = {
            let cas = SqliteCas::open(dir.db()).unwrap();
            let hash = cas.put(br#"[{"id":0},{"id":1},{"id":2}]"#);
            cas.flush();
            hash
        };
        let server = Coffer::with_sqlite(dir.db()).unwrap();

        let bytes = server
            .get_handle(hash.short())
            .expect("unique sentinel prefix should resolve in shared SQLite CAS");
        assert_eq!(&bytes[..], br#"[{"id":0},{"id":1},{"id":2}]"#);
        let rows = render_json_rows(&bytes, Some(1), Some(1), DEFAULT_MAX_ROWS).unwrap();
        assert!(rows.contains(r#""id": 1"#), "{rows}");

        assert!(
            server.get_handle(&hash.as_str()[..7]).is_none(),
            "prefixes shorter than the shared-CAS sentinel floor must not resolve"
        );
    }

    #[test]
    fn render_json_rows_returns_a_small_window_with_offsets() {
        let rows = br#"[{"id":0},{"id":1},{"id":2},{"id":3}]"#;
        let got = render_json_rows(rows, Some(1), Some(2), DEFAULT_MAX_ROWS).unwrap();
        assert!(got.starts_with("rows 1..3 of 4 (1 before, 1 after)\n"));
        assert!(got.contains(r#""id": 1"#), "{got}");
        assert!(got.contains(r#""id": 2"#), "{got}");
        assert!(!got.contains(r#""id": 0"#), "{got}");
        assert!(!got.contains(r#""id": 3"#), "{got}");
    }

    #[test]
    fn render_json_rows_rejects_non_arrays() {
        let err = render_json_rows(br#"{"id":1}"#, None, None, DEFAULT_MAX_ROWS).unwrap_err();
        assert_eq!(err, "held data is JSON but not an array");
    }

    #[test]
    fn render_json_rows_clamps_oversized_limit_to_max_rows() {
        // A 50-element array with a deliberately huge requested limit: the page must be clamped to
        // max_rows so coffer_rows cannot re-bloat the model context, while `total` still reports 50.
        let items: Vec<String> = (0..50).map(|i| format!(r#"{{"id":{i}}}"#)).collect();
        let array = format!("[{}]", items.join(","));
        let got = render_json_rows(array.as_bytes(), None, Some(usize::MAX), 10).unwrap();
        assert!(
            got.starts_with("rows 0..10 of 50 (0 before, 40 after)\n"),
            "{got}"
        );
        assert!(got.contains(r#""id": 9"#), "{got}");
        assert!(!got.contains(r#""id": 10"#), "{got}");
    }

    #[test]
    fn render_json_path_selects_nested_value() {
        let json = br#"{"items":[{"name":"alpha"},{"name":"beta"}],"count":2}"#;
        let got = render_json_path(json, "$.items[1].name").unwrap();
        assert_eq!(got, "json path $.items[1].name\n\"beta\"");
    }

    #[test]
    fn render_json_path_reports_missing_field() {
        let err = render_json_path(br#"{"items":[]}"#, "$.missing").unwrap_err();
        assert_eq!(err, "path field not found: missing");
    }

    #[test]
    fn render_text_lines_returns_numbered_range() {
        let got =
            render_text_lines(b"one\ntwo\nthree\nfour\n", Some(2), Some(3), None, None).unwrap();
        assert_eq!(
            got,
            "lines 2..3 of 4 (1 before, 1 after)\n     2|two\n     3|three"
        );
    }

    #[test]
    fn render_text_lines_returns_tail() {
        let got = render_text_lines(b"one\ntwo\nthree\nfour\n", None, None, None, Some(2)).unwrap();
        assert_eq!(
            got,
            "lines 3..4 of 4 (2 before, 0 after)\n     3|three\n     4|four"
        );
    }

    #[test]
    fn render_text_lines_rejects_conflicting_selectors() {
        let err = render_text_lines(b"one\n", Some(1), None, Some(1), None).unwrap_err();
        assert_eq!(err, "choose head/tail or start_line/end_line, not both");
    }

    #[test]
    fn render_text_search_returns_matching_line_numbers() {
        let got =
            render_text_search(b"INFO start\nError failed\ninfo retry\n", "error", None).unwrap();
        assert_eq!(
            got,
            "1 matches for \"error\" in 3 lines (showing 1, 0 omitted)\n     2|Error failed"
        );
    }

    #[test]
    fn render_text_search_limits_and_reports_omitted_matches() {
        let got = render_text_search(b"error a\nok\nerror b\nerror c\n", "error", Some(2)).unwrap();
        assert_eq!(
            got,
            "3 matches for \"error\" in 4 lines (showing 2, 1 omitted)\n     1|error a\n     3|error b"
        );
    }

    #[test]
    fn render_text_search_reports_no_matches() {
        let got = render_text_search(b"alpha\nbeta\n", "gamma", None).unwrap();
        assert_eq!(got, "0 matches for \"gamma\" in 2 lines");
    }

    #[test]
    fn sqlite_handle_store_survives_new_server_instance() {
        let dir = TempDir::new("persistent-handles");
        let handle = {
            let server = Coffer::with_sqlite(dir.db()).unwrap();
            let handle = server
                .put_bytes(b"persistent stderr\nerror[E0308]\n")
                .as_str()
                .to_string();
            assert_eq!(
                server.get_handle(&handle).as_deref(),
                Some(&b"persistent stderr\nerror[E0308]\n"[..])
            );
            handle
        };

        let reopened = Coffer::with_sqlite(dir.db()).unwrap();
        assert_eq!(
            reopened.get_handle(&handle).as_deref(),
            Some(&b"persistent stderr\nerror[E0308]\n"[..])
        );
    }

    #[test]
    fn sqlite_handle_store_creates_parent_directory() {
        let dir = TempDir::new("sqlite-parent");
        let db = dir.0.join("nested").join("session.db");

        let server = Coffer::with_sqlite(&db).unwrap();
        let handle = server.put_bytes(b"parent dir created").as_str().to_string();

        assert!(db.exists());
        assert_eq!(
            server.get_handle(&handle).as_deref(),
            Some(&b"parent dir created"[..])
        );
    }

    #[test]
    fn status_reports_memory_store_metrics() {
        let server = Coffer::in_memory();
        server.put_bytes(b"abc");
        server.put_bytes(b"defg");

        let status = server.status_text();

        assert!(status.contains("store: memory"), "{status}");
        assert!(status.contains("handles: 2"), "{status}");
        assert!(status.contains("resident_bytes: 7"), "{status}");
        assert!(status.contains("retrieve_default_bytes:"), "{status}");
        assert!(status.contains("retrieve_max_bytes:"), "{status}");
    }

    #[test]
    fn status_reports_sqlite_durability_metrics() {
        let dir = TempDir::new("sqlite-status");
        let server = Coffer::with_sqlite(dir.db()).unwrap();
        server.put_bytes(b"sqlite status");

        let status = server.status_text();

        assert!(status.contains("store: sqlite"), "{status}");
        assert!(status.contains("handles: 1"), "{status}");
        assert!(status.contains("resident_bytes: 13"), "{status}");
        assert!(status.contains("sqlite_db_bytes:"), "{status}");
        assert!(status.contains("sqlite_wal_bytes:"), "{status}");
        assert!(status.contains("sqlite_shm_bytes:"), "{status}");
        assert!(status.contains("sqlite_total_bytes:"), "{status}");
        assert!(status.contains("soft_cap_bytes:"), "{status}");
        assert!(status.contains("resident_cap_bytes:"), "{status}");
        assert!(status.contains("resident_evictions: 0"), "{status}");
        assert!(status.contains("warm_bytes_on_open: false"), "{status}");
        assert!(status.contains("trust_hashes_on_open: false"), "{status}");
        assert!(status.contains("checkpoint_every_blobs:"), "{status}");
        assert!(status.contains("wal_checkpoints:"), "{status}");
        assert!(status.contains("wal_checkpoint_failures: 0"), "{status}");
        assert!(status.contains("durability_lag: 0"), "{status}");
        assert!(status.contains("dropped_writes: 0"), "{status}");
        assert!(status.contains("persisted_blobs_this_run: 1"), "{status}");
        assert!(status.contains("retrieve_default_bytes:"), "{status}");
        assert!(status.contains("retrieve_max_bytes:"), "{status}");
    }

    #[test]
    fn positive_usize_parser_ignores_zero_or_bad_values() {
        assert_eq!(positive_usize_from_value(Some("1")), Some(1));
        assert_eq!(positive_usize_from_value(Some(" 4096 ")), Some(4096));
        assert_eq!(positive_usize_from_value(Some("0")), None);
        assert_eq!(positive_usize_from_value(Some("bad")), None);
        assert_eq!(positive_usize_from_value(None), None);
    }

    #[test]
    fn run_policy_disabled_by_default_and_enables_on_truthy() {
        // Off unless explicitly enabled — the kill-switch.
        assert!(run_policy_from_values(None, None).permits("ls").is_err());
        assert!(
            run_policy_from_values(Some("0"), None)
                .permits("ls")
                .is_err()
        );
        assert!(
            run_policy_from_values(Some("false"), None)
                .permits("ls")
                .is_err()
        );
        assert!(
            run_policy_from_values(Some("1"), None)
                .permits("ls")
                .is_ok()
        );
        assert!(
            run_policy_from_values(Some("YES"), None)
                .permits("anything goes")
                .is_ok()
        );
    }

    #[test]
    fn run_policy_allowlist_restricts_to_prefixes() {
        let p = run_policy_from_values(Some("1"), Some("kubectl, git "));
        assert!(p.permits("kubectl get pods").is_ok());
        assert!(p.permits("  git status").is_ok()); // leading whitespace trimmed before matching
        assert!(p.permits("rm -rf /").is_err()); // not on the allowlist
        assert!(p.permits("github --help").is_err()); // prefix must end at a word boundary
        assert!(p.permits("git status; cat ~/.ssh/id_rsa").is_err());
        assert!(p.permits("git status && cat ~/.ssh/id_rsa").is_err());
        assert!(p.permits("git status | cat").is_err());
        let subcommand = run_policy_from_values(Some("1"), Some("git status"));
        assert!(subcommand.permits("git status --short").is_ok());
        assert!(subcommand.permits("git statusx").is_err());
        // an allowlist that is only separators/space imposes no restriction (enabled still required)
        let empty = run_policy_from_values(Some("1"), Some(" , "));
        assert!(empty.permits("rm -rf /").is_ok());
        // allowlist without enable is still refused (enable is the primary guard)
        assert!(
            run_policy_from_values(None, Some("kubectl"))
                .permits("kubectl get pods")
                .is_err()
        );
    }

    #[test]
    fn run_limit_parser_uses_defaults_and_positive_overrides() {
        assert_eq!(
            run_limits_from_values(Some("7"), Some("2")),
            RunLimits {
                timeout_seconds: 7,
                max_output_bytes: 2 * super::MIB,
            }
        );
        assert_eq!(
            run_limits_from_values(Some("0"), Some("bad")),
            RunLimits {
                timeout_seconds: super::DEFAULT_RUN_TIMEOUT_SECONDS,
                max_output_bytes: super::DEFAULT_MAX_RUN_OUTPUT_BYTES,
            }
        );
    }

    #[tokio::test]
    async fn run_shell_command_times_out_and_returns_partial_capture() {
        let capture = run_shell_command(
            "printf before; sleep 2; printf after",
            RunLimits {
                timeout_seconds: 1,
                max_output_bytes: 1024,
            },
        )
        .await
        .unwrap();

        assert!(capture.timed_out);
        assert!(!capture.output_truncated);
        assert_eq!(capture.status, None);
        assert_eq!(String::from_utf8_lossy(&capture.bytes), "before");
    }

    #[tokio::test]
    async fn run_shell_command_enforces_output_cap() {
        let capture = run_shell_command(
            "i=0; while [ $i -lt 20 ]; do printf 0123456789; i=$((i + 1)); done",
            RunLimits {
                timeout_seconds: 10,
                max_output_bytes: 64,
            },
        )
        .await
        .unwrap();

        assert!(!capture.timed_out);
        assert!(capture.output_truncated);
        assert_eq!(capture.status, None);
        assert_eq!(capture.bytes.len(), 64);
    }

    #[test]
    fn retrieve_limits_parser_uses_defaults_and_caps_default_to_max() {
        assert_eq!(
            retrieve_limits_from_values(None, None),
            RetrieveLimits {
                default_bytes: 64 * 1024,
                max_bytes: 1024 * 1024
            }
        );
        assert_eq!(
            retrieve_limits_from_values(Some("20"), Some("10")),
            RetrieveLimits {
                default_bytes: 10,
                max_bytes: 10
            }
        );
        assert_eq!(
            retrieve_limits_from_values(Some("bad"), Some("30")),
            RetrieveLimits {
                default_bytes: 30,
                max_bytes: 30
            }
        );
    }
}
