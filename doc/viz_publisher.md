# 歩容ライブ可視化 — 配信側の実装契約

`viz::GaitVizFrame` を Zenoh で流して可視化する側（articara の Live feed など）が
期待する、**配信側**の契約。ここに書いたとおりに送れば、実測を不透明・指令を半透明の
ゴーストで重畳する表示が成立する。

コード上の正典は [`src/viz.rs`](../quadruped-gait/src/viz.rs) のモジュールドキュメント
（`cargo doc -p quadruped-gait --features viz --open`）。相違があればそちらが正で、
この文書は運用上の注意を足したもの。

## 0. 依存

```toml
quadruped-gait = { git = "https://github.com/takarakasai/quadruped-gait.git", features = ["viz"] }
```

- 必要 rev: **`1500acc`** 以降（`GaitVizFrame::pose_rp` と `VIZ_KEY_MEASURED` を含む）
- `viz` feature で `GaitVizFrame` に serde derive が付く。zenoh は自前で持つ（`zenoh = "1.9"`）

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
  サンプリングするため、これが両者のズレを1配信周期に抑える唯一の保証になる
- 配信レートは制御周期から間引く（既定 100 Hz 相当）。500 Hz 制御なら5 tick に1回
- **最初の状態読み戻しが済むまで measured を送らない。** ゼロ姿勢のフレームは
  「崩れ落ちたロボット」として描画される

## 4. 制御ループを止めないこと

- `session.put(..).wait()` は**ブロッキングのネットワーク呼び出し**。制御ループ内で
  直接呼ばない。JSON 直列化も同様
- 推奨形：`sync_channel(8)` 程度の**有界チャネル**で publisher スレッドへ渡し、
  満杯なら `try_send` の失敗として**捨てる**（可視化は lossy でよい）。
  取りこぼし数を数えて終了時に出すと、詰まりが「健全な配信」に見えなくなる
- 参考実装: [go2-gait-runner](https://github.com/takarakasai/go2-gait-runner) の
  `mod viz_pub`（`VizPublisher::new` がスレッドを立て、`publish()` はフレーム構築と
  `try_send` のみ）

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

### 5.2 明示している設定（唯一の非既定）

multicast が使えない環境（同一ホスト / WSL2 / multicast 不可の LAN）向け:

- 配信側: `listen/endpoints = ["tcp/0.0.0.0:7447"]`、`scouting/multicast/enabled = false`
- 受信側: `connect/endpoints = ["tcp/<host>:7447"]`、`scouting/multicast/enabled = false`

peer の listen 既定がエフェメラルポートなので、**接続先を固定したいなら明示が必須**。
配信元が複数ある場合は別ポートを割り当てる（7447, 7448, …）。受信側 articara は
ストリームごとに接続先を指定できる。multicast が通る LAN なら両側とも無指定でよい。

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
