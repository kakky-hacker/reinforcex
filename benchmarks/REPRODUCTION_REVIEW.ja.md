# CPUベンチマーク再現手順のレビュー

2026-09-19 JST。学習中の主行列は中断せず、core/FFI/examples、凍結4script、元99runのmanifestを変更しない範囲で確認した。変更対象は[README](README.md)と[再現手順](REPRODUCTION.md)。別担当の新規`prepare_reproduction.py`ともCLIの意味を照合した。

## 修正した説明

- 主90runとTianshou補助9runを区別し、全99run・141 workerモデル・119,193,600訓練transitionの構成を明記した。
- 元manifestを直接起動する例を廃止し、既定dry-runの準備CLIで新規outputへパスを置換したコピーを作る手順へ変更した。Pythonだけでなくrunner、config、output、FFI library、loaderのパスが対象。
- native LibTorch 2.7とSB3 Torch 2.14、Tianshou Torch 2.7を別venv/別processで扱う。nativeのloader環境変数は親shellへ恒久exportせず、native子processへ限定する。
- Cargoの`LIBTORCH`はdistribution root、準備CLIの`--libtorch-dir`はshared librariesがある`lib/`という違いを明記した。
- 依存ファイルは測定macOS環境のpackage-version snapshotで、全OS対応のwheel lockではないこと、Box2D source buildのcompiler/SWIG/pygame要件、プロット依存のあるPythonを説明した。
- `final.json`存在によるlauncherのskipは監査成功を意味しないこと、同一manifestへの二重起動に排他制御がないこと、部分runをweightsだけで継続できないことを説明した。
- transition budgetとepisode数、全worker合計budget、raw/学習整形報酬、評価episode SDと訓練seed SDを区別し、任意outputを明示する監査・集計・再ロード・RND検証コマンドを追加した。

## 実測した手順の検証

1. 一時ディレクトリへbase commit `757cca7df380113e7eefdc717b9b8ba21b6ff531`を展開し、`source_worktree.patch`の適用可否を確認後に適用した。`source_snapshot_before.json`対象52ファイルのSHA-256は全件一致した。本作業treeのソースは変更していない。
2. 準備CLIを実manifest・実Pythonパス・実libraryパスでdry-runし、出力directoryが作られないことを確認した。次に別の一時outputで`--write`を実行し、99runのローカルmanifest/config/snapshotコピーを生成した。学習は起動していない。
3. その空の新規outputに対してmain audit、main renderer（`--no-plots`）、stability、distribution、checkpoint reload検証を明示pathで実行した。すべて正常終了し、元の完了済み結果を混入しなかった。

|確認|期待した結果|実測|
|---|---|---|
|Main audit|90 pending / invalid 0|一致|
|Main renderer|90 runs / 33 groups / 完了0|一致、discovery/run error 0|
|Stability|90 runs / 132 models / 完了0|一致|
|Distribution|99 runs / 141 modelsすべてpending|一致、error 0|
|Checkpoint reload|132 modelsすべてpending|一致、推論episode 0|

一時生成物は検証後に削除した。学習run/推論workerは起動していない。検証後、凍結4scriptのSHA-256は保存snapshotと全件一致した。ドキュメント内の相対リンクにも欠落はない。

## 残る制約

`prepare_reproduction.py`はsource/configuration一致とパスの存在を確認する準備ツールで、Python packageの実version、LibTorch binary version/architecture、実際のloader動作までは実行検証しない。別machineでの依存install/buildそのものは今回再実施していない。

任意outputで全99runを学習し、main90の監査・図、99runの報酬分布を作ることは可能。一方Ti専用auditのsnapshot読取、Ti比較図・checkpoint検証の一部、campaign statusはcanonical `reports/oss_benchmarks` を入力として残している。READMEで`--output`だけ変えて新結果を読めるとは説明していない。専用の分析checkoutへprepared manifest/新結果を配置するか、これら非凍結の補助ツールのpath引数追加を別途レビューする必要がある。

旧`finalize_results.py`は93runを待つ補助scriptで、Ti Double DQN 6runや後発の分布・再ロード検証を網羅しない。このscriptの成功を99run全体の完了証明には使えない。既存の実行中watcherは変更していない。
