# 歩容ライブ可視化 — 配信側の実装契約

`viz::GaitVizFrame` を Zenoh で流して可視化する側（articara の Live feed など）が
期待する、**配信側**の契約。ここに書いたとおりに送れば、実測を不透明・指令を半透明の
ゴーストで重畳する表示が成立する。

コード上の正典は [`src/viz.rs`](../quadruped-gait/src/viz.rs) のモジュールドキュメント
（`cargo doc -p quadruped-gait --features viz --open`）。相違があればそちらが正で、
この文書は運用上の注意を足したもの。

## 0. 依存

```toml
# 送信の仕組みごと使う（推奨）
quadruped-gait = { git = "https://github.com/takarakasai/quadruped-gait.git", features = ["viz-pub"] }
# 型だけ使って自前で送る場合
quadruped-gait = { git = "https://github.com/takarakasai/quadruped-gait.git", features = ["viz"] }
```

- `viz-pub` feature が [`viz_pub::VizPublisher`] を提供する。**トランスポート・スレッド・
  背圧・2ストリームの対応付けはこれが持っている**ので、§3 と §4 の要求は自動的に満たされる。
  配信側が書くのは「フレームをどう組むか」（関節順・符号・実測姿勢の出所）だけ
- `viz` feature だけなら `GaitVizFrame` の serde derive のみ。zenoh は自前で持つ
  （`zenoh = "1.9"`）。この場合は §3〜§5 を自力で満たすこと
- 必要 rev: **`5044caa`** 以降（`viz_pub` / `viz_net` を含む rev は §5.2 参照）

## 1. 2ストリーム

| 用途 | キー定数 | 既定値 | 中身 |
|---|---|---|---|
| 指令 | `viz::VIZ_KEY_PLANNED` | `go2/gait/planned` | コントローラが出した目標 |
| 実測 | `viz::VIZ_KEY_MEASURED` | `go2/gait/measured` | ロボットから読み戻した実状態 |

- **キーは必ず別にする。** チャネルは latest-wins なので、1本に両方流すと上書きし合い、
  受信側は指令と実測の間でガタつく。受信側は同一キー設定を検出したら measured を落とす
- 実測を持たない配信元（オフライン再生等）は planned のみでよい。受信側は planned 単独なら
  それでモデルを駆動し、ゴーストを描かない
- 逆に measured のみでも成立する

キーの構造は **`<robot>/gait/<stream>`**。先頭チャンクがロボットの識別子で、機体は
これで切り分ける（既定値の `go2` は既にこの規約に従っている）。詳細は §5.4。

## 2. フレーム内容

`GaitVizFrame`（JSON、`Encoding::APPLICATION_JSON` で put）:

| フィールド | 型 | 要求 |
|---|---|---|
| `version` | u32 | `VIZ_FORMAT_VERSION`。受信側は不一致フレームを捨てる |
| `seq` | u64 | 単調増加。**planned と measured の対で同じ値**（下記） |
| `t_s` | f64 | 走行開始からの秒 |
| `pose` | [f64;4] | 胴体の world `[x, y, z, yaw]`（m, rad）。z は接地面からの胴体高さ |
| `pose_rp` | [f64;2] | 胴体の `[roll, pitch]`（rad）。水平なら `[0,0]` |
| `joints` | [f64;12] | slot 順 **FL, FR, RL, RR** × (hip, thigh, calf) |
| `stance` | [bool;4] | 同 slot 順の接地フラグ |

### 関節角の規約（最頻出の事故）

- `GaitVizFrame::from_output()` は**IK 規約**で埋める。実機に送るのと同じ
  `joint_signs` の符号補正を**かけてから** publish する。忘れると膝が逆に曲がって描画される
- ロボットから読み戻した角度（`LowState.motor_state[].q` 等）は**すでにモデル規約**なので
  符号補正は不要。**slot 順への並べ替えだけ**行う
  （Go2 モータ順 FR/FL/RR/RL → slot FL/FR/RL/RR のベース index は `[3, 0, 9, 6]`）

### 姿勢の規約

- measured の `pose` は**実測値を入れる**。指令姿勢を入れると2体が構造的に重なり、
  絵は綺麗だがロボットが実際どこにいるかが消える。重ねたい受信側は自分でアンカーし直す
- measured の `pose_rp` は IMU の roll/pitch をそのまま。planned は水平計画なので `[0,0]`
  （MPC 等で計画姿勢を持つなら入れてよい）
- 位置 `x, y` の出所は問わない（接地脚オドメトリ積分、KF、外部計測いずれも）。
  受信側は「x/y はドリフトしうる推定」「z と roll/pitch は直読み」という前提で
  アンカーの既定を決めている

## 3. タイミング

- **planned と measured は同一 tick・同一 `seq` で送る。** 受信側は2ストリームを独立に
  サンプリングするため、これが両者のズレを1配信周期に抑える唯一の保証になる。
  `VizPublisher::publish()` は両フレームの `seq` を自分で打ち直してこれを保証する
  （手で合わせるには壊れやすすぎるため）
- 配信レートは制御周期から間引く（既定 100 Hz 相当）。500 Hz 制御なら5 tick に1回
- **最初の状態読み戻しが済むまで measured を送らない。** ゼロ姿勢のフレームは
  「崩れ落ちたロボット」として描画される

## 4. 制御ループを止めないこと

- `session.put(..).wait()` は**ブロッキングのネットワーク呼び出し**。制御ループ内で
  直接呼ばない。JSON 直列化も同様
- 推奨形：`sync_channel(8)` 程度の**有界チャネル**で publisher スレッドへ渡し、
  満杯なら `try_send` の失敗として**捨てる**（可視化は lossy でよい）。
  取りこぼし数を数えて終了時に出すと、詰まりが「健全な配信」に見えなくなる
- **`viz_pub::VizPublisher` がこれを実装済み**。`publish()` は間引き判定 →
  （publish する tick だけ）フレーム構築 → 有界チャネルへ `try_send`、で終わる。
  セッションを持つスレッドが直列化と put を担当する。取りこぼし数は `dropped()`
- 自前で書く場合も同じ形にすること。参考は
  [go2-gait-runner](https://github.com/takarakasai/go2-gait-runner) の呼び出し側

## 5. Zenoh トランスポート仕様

実装（`go2-gait-runner` の `viz_pub`、`quadruped-gait` の `viz_sub`）は zenoh 1.9 を
**ほぼ既定設定のまま**使っている。以下の既定値は zenoh 1.9 のソースで確認したもので、
明示的に設定しているのは §5.2 の endpoint 関連だけ。

### 5.1 セッション

| 項目 | 値 | 出所 |
|---|---|---|
| モード | `peer` | `zenoh_config::defaults::mode` |
| scouting multicast | 有効、`224.0.0.224:7446`、timeout 3000 ms | `defaults::scouting::multicast` |
| peer の listen 既定 | `tcp/[::]:0`（**エフェメラルポート**） | `ListenConfig::default()` |
| connect の再試行 | peer は `timeout_ms = -1`（無限）、`exit_on_failure = false` | `defaults::connect` |
| 再試行間隔 | 1000 ms から開始、×2 で増加、上限 4000 ms | `ConnectionRetryModeDependentConf::default()` |

再試行が無限なので **受信側を先に起動してもよい**。配信側が上がれば最大4秒ほどで繋がる。

### 5.2 トポロジ（接続の向き）

zenoh の peer は対称なので、**どちらが待ち受けてもよい**。`viz_net::VizEndpoints` が
その選択を持ち、配信側 (`viz_pub`) と受信側 (`viz_sub`) の両方が同じ型を受け取る。

| トポロジ | 配信側 | 受信側 | 使う場面 |
|---|---|---|---|
| discovery | `auto()` | `auto()` | multicast が通る LAN。アドレス不要 |
| 配信が待つ | `listen(["tcp/0.0.0.0:7447"])` | `connect(["tcp/<robot>:7447"])` | 既定。ロボットのアドレスが既知 |
| 受信が待つ | `connect(["tcp/<pc>:7447"])` | `listen(["tcp/0.0.0.0:7447"])` | **ロボット側のアドレスが動く / NAT 越し** |
| router 経由 | `connect(["tcp/<router>:7447"])` | `connect(["tcp/<router>:7447"])` | zenohd を挟む。多対多になる場合 |

- `listen` と `connect` は排他ではない。両方指定して「受信を待ちつつ router にも繋ぐ」も可
- エンドポイントを1つでも指定すると **multicast scouting は自動的に無効**になる
  （指定する理由は大抵 discovery が効かないため）。`with_multicast(true)` で上書き可
- peer の listen 既定がエフェメラルポートなので、**待ち受け側は明示が必須**
- 配信元が複数ある場合は別ポートを割り当てる（7447, 7448, …）
- `go2-gait-runner` では `--viz-endpoint`（listen）と `--viz-connect`（connect）。
  どちらもカンマ区切りで複数指定できる
- articara では Live feed 窓の `mode`（auto / connect / listen）と `endpoint`

**受信側が listen する場合、セッションは1本にすること。** planned と measured で別々に
セッションを開いて同じポートを listen すると2本目が bind に失敗する。`viz_sub::VizSession`
を1つ開いて `subscribe()` を2回呼ぶ（articara はそうしている）。配信元が2つに分かれて
いる場合だけセッションも2本になる。

### 5.3 QoS

`session.put()` に QoS を指定していないので、すべて zenoh の既定値。

| 項目 | 既定値 | 意味 |
|---|---|---|
| CongestionControl | `Drop` | **送信キューが詰まったらブロックせず捨てる**。可視化の要求と一致 |
| Priority | `Data` | 8段階の中位 |
| Reliability | `Reliable` | トランスポート層の再送あり。ただし輻輳時は上の `Drop` が優先で捨てられる |
| express | `false` | バッチング有効。スループット優先で、遅延はバッチ分だけ増える |

`Drop` なので **受信側が固まっても配信側が無限にブロックすることはない**。
それでも put を制御ループから出すべき理由は「無限停止の回避」ではなく、直列化・確保・
ロック・write システムコールが 500 Hz の tick に乗る**ジッタ**として効くため（§4）。

変える必要が出るとしたら:

- 可視化が他のトラフィックを圧迫する → `Priority::Background`
- 1フレームの遅延を削りたい → `express = true`（バッチングを切る。帯域効率は落ちる）

### 5.4 キー空間

```
<robot>/gait/<stream>
   |      |      +--- planned | measured
   |      +---------- 用途。将来 gait 以外を流すならここで分ける
   +----------------- ロボット識別子。機体はここで切り分ける
```

- `<robot>` は**機体1台に1つ**。同型機が複数あるなら型名では足りないので
  `go2-01`, `go2-02` のように識別子まで含める
- 使える文字: zenoh のキー式チャンクなので `/ * ? # $` が使えない（`#` と `?` は禁止、
  `$` は `$*` の形しか許されずそれはワイルドカード）。**`[a-z0-9][a-z0-9-]*` に収めるのが安全**
- 既定の `go2/gait/planned` は既にこの規約に従っている（`go2` が `<robot>`）ので、
  規約の導入で既存の設定は壊れない
- 配信側はロボット名だけ差し替えられるようにする（`go2-gait-runner` は `--viz-robot NAME`。
  キー全体を `--viz-key` で指定する経路も残っている）

**ワイルドカード購読はしないこと。** `*/gait/measured` のような購読は zenoh 的には可能だが、
受信側 articara は1つのモデルを1つのフレーム列で駆動するので、複数機体のフレームが同じ
モデルを奪い合って姿勢が飛ぶ。**1窓につき1機体を明示指定**する。複数機体を同時に見たい
場合は articara を複数起動する。

### 5.5 ペイロードと帯域

JSON で put（`Encoding::APPLICATION_JSON`）。1フレームの実測値:

| ケース | サイズ |
|---|---|
| 全ゼロ（下限） | 168 B |
| 実走の典型値（f64 が長い展開になる） | 324 B |
| 最悪ケース（全フィールドが最長展開） | 508 B |

帯域（1機体あたり、TCP / Zenoh のヘッダを除く）:

| レート | planned のみ | planned + measured |
|---|---|---|
| 100 Hz（既定） | 32 KiB/s | **63 KiB/s**（最悪 99 KiB/s） |
| 50 Hz | 16 KiB/s | 32 KiB/s |
| 30 Hz | 9.5 KiB/s | 19 KiB/s |

LAN では無視できる量。無線経由や機体を多数並べる場合はレートを落とす。JSON をやめて
bincode 等にすれば3〜4割減るが、可読性を失うので現状は JSON のまま。

### 5.6 時刻

- フレームの `t_s` は**走行開始からの経過秒**。壁時計ではなく、機体間で同期もしていない
- zenoh の timestamp 機能（`put().timestamp()`）は使っていない
- 送受信でホストが違っても時刻同期は不要。受信側は `t_s` を表示に使わず、
  2ストリームの対応には `seq` だけを使う

## 6. 受信側の挙動（送信側が知っておくべき分）

- measured が来た時点でそちらがモデルを駆動し、planned は半透明ゴーストになる
- measured が途切れると**最終値でラッチ**して更新が止まる（planned に勝手に戻らない）。
  再開時は最新フレームから再開（滞留の早送りはしない）
- ゴーストの胴体位置は3モードで切替可能。既定は「x, y, yaw は measured に合わせ、
  高さと姿勢は指令のまま」。よって **z と roll/pitch の実測値は画面上で誤差として見える**

## 7. 検証手順（実機なしで可）

1. 配信側を `--viz-endpoint tcp/0.0.0.0:7447` 相当で起動
2. 受信側（articara）を `cargo run --features viz -- models/unitree_go2/go2.misa` で起動
3. Live feed (Zenoh) 窓 → endpoint に `tcp/127.0.0.1:7447` → Subscribe
4. `● target — frame #N` と `● measured — frame #N` が両方増えれば経路 OK
5. anchor を `full` にして関節差、`world` にして胴体差が見えることを確認

## 8. 未対応・要相談

- **速度／接触力**は wire に無い。GRF 可視化まで欲しくなったら `GaitVizFrame` の追加拡張
  （`#[serde(default)]` 付きの追加フィールドならバージョン据え置きで前方後方互換）
- planned 側の `pose_rp` は現状 `[0,0]` 固定（`BodyState` が位置と yaw しか持たないため）。
  MPC の計画姿勢を載せるなら送信側で埋める
