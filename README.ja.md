# coffer

[![ci](https://github.com/haru0416-dev/coffer/actions/workflows/ci.yml/badge.svg)](https://github.com/haru0416-dev/coffer/actions/workflows/ci.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)

[English](README.md) | **日本語**

> ツールの結果がモデルに読み切れない大きさなら、切り捨てるのではなく **保持する**。
> coffer は元のバイト列を content store に保管し、モデルには検証可能な handle を渡し、
> count / sum / max / group-by / join の質問に **全データ**に対して正確に答えます。

![coffer-wrap が 20,000 pod のツール結果を照会可能な handle に変える様子: elide された中央の needle への exact な回答、provenance 付きの count、sha256 検証付き retrieve](demo/wrap.gif)

**Status: experimental.** エンジン、3 つの surface(MCP gateway・MCP server・transparent proxy)、
production 向けの smoke gate は実在し、テストされています。今日の時点で機械的に検証可能な性質は
2 つ — byte-exact な復元と exact な集計です。end-task の *accuracy* に関する主張(tool-output の
圧縮はモデルの回答を保つのか、改善するのか?)は、結果より先に固定したプロトコルに従って扱い、
圧縮率で語ることはありません。

## 問題

`kubectl get pods -o json` 1 回、CI の job log 1 本、API のデフォルト設定での issue 一覧 1 回 —
たった 1 回のツール呼び出しが数万〜数十万 token を返すことは日常的にあります。MCP host はツール
出力に上限を設けており(一般に 25k token 前後)、その呼び出しは**そのまま失敗する**か、黙って
切り捨てられるか、context を溢れさせて以降の回答すべてを劣化させます(いわゆる "context rot")。
そして既存の対処はどれも情報を捨てます — head/tail の窓は「埋もれた答えが載っている行」をこそ
落とし、lossy な要約は元のバイト列を二度と返せません。

## coffer がやること

coffer は payload を **1 バイトも失わずに** context の外へ移し、*計算のほう*をバイト列のある場所へ
移します:

- **Byte-exact な可逆性。** offload されたバイト列は SHA-256 content-addressed store に保管され、
  モデルには短い handle / fact card が見えます。`reconstruct(compress(x)) == x` が byte 単位で成立 —
  これは Stage-0 の不変条件で、property-test され、**読み出しのたびに hash と照合されます**。
  store の破損は「黙って間違ったバイト列」ではなく hard error になります。落とした needle は
  失われず、回収可能です。
- **Exact な compute-digest。** 生き残ったどの行にも答えが載っていない質問があります —
  「エラーは何件?」「最も restart している pod は?」「p95 latency は?」。coffer はこれらを
  **offload 済みのバイト列を含む全データに対して、Rust で正確に**計算し、すべての数値に
  裏付け行の index(provenance)を添えて返します。契約は refuse-rather-than-guess: 型が混在する
  field は値をこっそり読み飛ばすのではなく、クエリ自体を拒否します。frontier モデルに自分の
  context 内の数千行を count / sum させると間違えますが、保持したバイト列の上で Rust が計算すれば
  間違えません。
- **Verifier。** `coffer_check_claim` はエージェントが*主張した*数値を再計算して AGREE/DISAGREE を
  裏付け行とともに返します。`coffer_receipt` / `coffer_verify_receipt` は可搬な exactness receipt
  (クエリ + 値 + 行 index + 裏付け行の SHA-256)を発行し、モデルなしで再実行・検証できます —
  実行可能なデモは `cargo run -p coffer-core --example verify`。

## Surfaces — 1 つの content store、3 つの入口

- **MCP gateway(`coffer-wrap`)— 1 行で導入できる本命の入口。** 既存の任意の stdio MCP server を
  包みます: `coffer-wrap -- <command>`。大きすぎるツール結果は byte-exact に offload され、
  fact card(handle + 正確な per-field 統計 + preview + 照会方法)に置き換わります。下記の照会
  ツール群が、包んだ server 自身のツールと並んで注入されます(collision-safe — 下流のツールを
  shadow することは決してありません)。host の出力上限で失敗するはずだった結果が、照会可能な
  handle になります。冒頭の gif がこの surface の一部始終です。
- **MCP server(`coffer-mcp`)。** 出力を読み込む代わりに、server 側に保持したまま照会するための
  ツール群。目的別に:
  - *正確に把握する*: `coffer_describe`(任意のレコード集合の行数 + per-field 統計/count-by)、
    `coffer_digest`(平易な英語で聞く exact な統計)、`coffer_aggregate`(述語付きの型付き
    `count|sum|mean|min|max`、provenance index 付き)、`coffer_bucket` / `coffer_window`
    (数値バンド / N 行ブロックごとのヒストグラム)、`coffer_join`(保持中の 2 データセットを
    server 側だけで semi-join / grouped join);
  - *読まずに絞り込む*: `coffer_query` / `coffer_select`(行をフィルタして新しい handle を得る —
    合成可能)、`coffer_pick`(provenance の行だけを取得して数値を再検証)、`coffer_search` /
    `coffer_lines` / `coffer_rows` / `coffer_json`(ログと JSON への有界な窓);
  - *検証する*: `coffer_check_claim`、`coffer_receipt`、`coffer_verify_receipt`(前述);
  - *回収・保持する*: `coffer_retrieve` / `coffer_unfold`(有界な byte 窓、hash 検証付き)、
    `coffer_ingest`(ファイルを保持)、`coffer_run`(シェルコマンドの出力を server 側で捕捉;
    `COFFER_MCP_ENABLE_RUN=1` でない限り無効)、`coffer_status`。
- **Transparent proxy(`coffer-proxy`)。** エージェントの base URL を向けるだけで、飛んでいく
  リクエスト内の大きなツール出力の値だけを書き換えます — Anthropic Messages の `tool_result`、
  OpenAI Responses の `*_call_output`、Ollama `/api/chat` の tool メッセージ。system/user/assistant
  の書かれたテキストと prompt-cache の prefix には決して触れません。書き換えた各ブロックの先頭には
  1 行の explainer が付き、`<<cof:…>>` マーカーが破損と誤解されることを防ぎます。fail-open:
  想定外のものはすべて素通しです。

3 つの surface は 1 つの content store を共有します。proxy が圧縮し、gateway が offload し、
MCP server が同じバイト列を回収・照会できます。

## 正直さを、先に

context 圧縮ツールの多くは、圧縮率と accuracy を別々のデータセットで測り、いちばん有利な条件だけを
報告します。coffer はその逆を、結果より先に固定したプロトコルとして約束します:

- **同一ワークロードでの end-task accuracy** を複数の圧縮率で測る → regime・content type ごとの
  accuracy-vs-compression 曲線。
- 決定的な比較は **同じ token 予算での naive な head/tail truncation との対決**。安い baseline が
  並ぶワークロードでは、並んだと明記する。
- 有利な裾ではなく、**負けうる典型的な regime** を報告する。
- token は**対象モデル自身の tokenizer**で数え、**retrieval の往復 token も数える**。byte 忠実な
  round-trip の検証は accuracy とは分けて報告する。

このうち 2 つの約束は**機械的で、API キーなしで再現できます**。`cargo run --release -p coffer-eval`
が、5,000 pod の `kubectl` dump(o200k で約 229k token)に対して以下の表を再生成します(token は
モデル自身の tokenizer でカウント):

| compression | byte-exact round-trip | coffer answer error | head/tail truncation error at the same budget (count · sum · argmax) |
|------------:|:---------------------:|:-------------------:|:---------------------------------------------------------------------|
| 33% | ✅ | **0.00%** | 33% · 34% · ❌ buried needle missed |
| 67% | ✅ | **0.00%** | 68% · 61% · ❌ buried needle missed |
| 87% | ✅ | **0.00%** | 87% · 83% · ❌ buried needle missed |
| 93% | ✅ | **0.00%** | 93% · 91% · ❌ buried needle missed |
| 97% | ✅ | **0.00%** | 98% · 96% · ❌ buried needle missed |

coffer の回答は(offload 済みも含む)**全バイト**に対して計算され、独立に算出した ground truth との
一致を assert しているため、全レベルで誤差 0.00% です。表の数値は**同じ token 予算**での
*truncation* baseline の誤差で、「窓に見えている行に対する完璧な集計」という甘めのモデル化です —
実際のモデルはこれ以上の行を見られない上に計算も苦手なので、これは baseline の上限です。exact な
回答の取得コストは dump のサイズによらず retrieval 約 20 token。truncation が**一様に悪いわけでは
ありません**: mean のようなサンプリングに頑健な統計では数%以内に収まります — coffer が勝つのは
「窓が落とした行に依存する答え」(count、sum、埋もれた極値)であり、harness は勝てない mean の列も
そのまま報告します。

エンジン自体も同じ流儀でデモしています — 5.7 MB の kubectl 形状の dump を実測 `o200k_base` で
2,170,329 → 216,857 token に削減し、byte 単位で再構築(`cmp` を画面上で実行):

![coffer が 5000 pod の kubectl dump を約 90% 圧縮し、それでも byte 単位で可逆である様子](demo/demo.gif)

残る約束 — **実モデルでの end-task accuracy** を、複数の圧縮率で、同じ truncation baseline と比べる
こと — は、上の harness では**決着しない**未解決の実験課題です: harness が証明するのは 2 つの
機械的性質(byte-exact round-trip と exact な集計)であって、「LLM の回答が良くなる」ことでは
ありません。accuracy の仮説が kill-probe で否定されたなら、その失敗した曲線も有用な公開結果です。

coffer が**勝たない**場面も同じだけ明確です。frontier モデルの context window に収まる普通の
retrieval では、入力を圧縮しても生のまま渡すのに勝てません — coffer は並ぶだけです。また
code 実行エージェントは自分でコードを書いて同じ exact な集計を計算できます — accuracy では
引き分けで、coffer の勝ちではありません。あえて言葉にする価値のある差分はもっと狭いものです:
coffer はバイト列がモデルに届く**前**の transport 層で動き、sandbox も codegen の往復も不要で、
元のバイト列をすべて回収可能に保ちます。

## Quickstart

```sh
# ビルド済みバイナリ(linux x86_64/aarch64, macOS x86_64/aarch64, windows x86_64):
#   https://github.com/haru0416-dev/coffer/releases — sha256 チェックサム付き。
# またはソースからインストール(Rust toolchain が必要):
cargo install --git https://github.com/haru0416-dev/coffer coffer-wrap --locked
# (coffer-mcp / coffer-proxy も同様)

# MCP gateway — context を溢れさせる server を包む(任意の stdio MCP server、任意の MCP host)。
# 実例: host の MCP 設定で、公式 filesystem server を包む場合:
#   "command": "coffer-wrap",
#   "args": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]

# MCP server — 出力を handle で保持し、正確に照会する:
coffer-mcp

# Transparent proxy — 飛んでいく tool_result ブロックを圧縮する:
coffer-proxy
ANTHROPIC_BASE_URL=http://127.0.0.1:8788   # COFFER_PROXY_UPSTREAM の既定は api.anthropic.com
```

どれにも `COFFER_CAS_DB=/path/to/cas.db` を設定すると、1 つの永続 store をプロセス間で共有できます —
gateway や proxy が offload したものを、MCP server が回収・照会できます。production 向けの配線は
[`docs/deployment.md`](docs/deployment.md) を参照。

npm launcher は [`npm/`](npm/) に scaffold があります: 公開されれば `npx coffer coffer-mcp` で
プラットフォーム別のビルド済みバイナリが動きます。**まだ npm registry には公開していません** —
当面は release バイナリを使うか、上記のとおりソースからビルドしてください。

デフォルトで安全側に倒しています: proxy は `COFFER_PROXY_ALLOW_PUBLIC=1` がない限り loopback 以外の
bind を拒否します(認証を持たず、upstream の key を中継するため)。MCP の `coffer_run` シェル
ツールは `COFFER_MCP_ENABLE_RUN=1` がない限り無効です。

## Layout

- `crates/` — エンジン(`coffer-core`)、content store(`coffer-cas`)、tokenizer-parity カウント
  (`coffer-tokenizer`)、MCP server(`coffer-mcp`)、MCP gateway(`coffer-wrap`)、transparent
  proxy(`coffer-proxy`)、再現可能ベンチマーク(`coffer-eval`、上の表 — `cargo run --release -p
  coffer-eval`)。
- [`docs/DESIGN.md`](docs/DESIGN.md) — 設計と仕様: 可逆性の不変条件、データモデル、圧縮
  パイプライン、budget search、compute-digest、surfaces、non-goals。
- [`docs/deployment.md`](docs/deployment.md) — MCP/proxy のデプロイ、shared-CAS の配線、制限事項。
- [`demo/`](demo/README.md) — 上の 2 本の gif。実コマンドから再現可能な方法で収録。

*このファイルは [README.md](README.md)(英語)の翻訳です。差異があれば英語版を正とします。*

## License

Apache-2.0。[`LICENSE`](LICENSE) を参照。
