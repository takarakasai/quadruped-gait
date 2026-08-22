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

## 5. Zenoh 設定

- 通常は multicast の自動探索でよい
- **同一ホスト / WSL2 / multicast 不可の環境**では、配信側が
  `listen/endpoints = ["tcp/0.0.0.0:7447"]` で待ち受け、`scouting/multicast/enabled = false`。
  受信側（articara）が `connect` する。配信元が複数ある場合は**別ポートで待ち受ける**こと
  （articara 側はストリームごとに接続先を指定できる）

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
