# 人手ラベル付き映像での評価手順

公式の [MOT17アーカイブ](https://motchallenge.net/data/MOT17/) から、
MOT17-02-SDP、04-SDP、09-SDPの3撮影場面を使用します。
[データセットの原論文](https://arxiv.org/html/2010.07548v2) の付録Cでは、
この3系列は固定カメラとされています。SDPの提供検出結果は使用せず、
元画像と人手の矩形・物体IDラベルだけを使います。
DPM/FRCNN/SDPは同じ撮影の複製なので、3倍の独立したデータとして数えません。

`mot17-plan-v1.json` のサンプリングを推論前に固定しました。
各系列8時点を等間隔に選び、各時点から1・15・60フレーム後を比較します。
30fpsなので、時間差は約0.033・0.5・2秒です。96画像・72ペアになります。
同じフレームを再利用し、同じ撮影内の画像も相関しています。

## 準備と元ラベルの固定

以下は明示的なネットワーク利用による素材準備です。
ZIPのHTTP Rangeを使い、選択した96画像と3系列のGT/設定ファイルだけを取得します。
アーカイブ全体5.86GBは取得しません。今回の選択エントリーの圧縮データは約20MBです。
サーバーがRangeを無視した場合は、全体を読まずに停止します。
元ファイルのCRC、SHA-256、アーカイブETagを記録・検証します。

```sh
.venv/bin/python scripts/prepare_mot17.py \
  --plan evaluation/mot17-plan-v1.json --output artifacts/mot17

.venv/bin/python scripts/build_mot17_manifest.py \
  --plan evaluation/mot17-plan-v1.json --assets artifacts/mot17 \
  --lock evaluation/mot17-assets-v1.json --output artifacts/mot17/new-manifest.json
```

今回の実行manifestは `artifacts/mot17/manifest-v1.json` です。
元画像、GT、変換済みのラベルはローカルの `artifacts/` に保存します。
wheel/sdistには素材・GT・変換済みラベルを含めません。
リポジトリ側にはサンプリング計画と検証用ハッシュを保存します。

## 評価対象と除外ルール

MOTの人物区分と、モデルのCOCO person区分は完全には一致しません。
[原論文のGT形式](https://arxiv.org/html/2010.07548v2) および
[TrackEvalのMOT用実装](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py)
を確認し、以下の診断用ルールを明示しています。

- GTのactiveフラグが非ゼロかつnative class=1の歩行者を、モデルのclass=0へ対応させます。
- class=2/7/8/12と、非activeのclass=1を明示的な除外対象として保存します。
  これらをモデルがpersonとして検出しても、一部は認識上の誤りとは限りません。
- 正解対象へ一対一で割り当てた後、残る予測が除外対象へIoU≥0.5で一対一に
  対応した場合、認識・対応の評価から除外します。除外対象一つで重複検出を
  すべて隠す処理はしません。IoUの感度確認でも除外側のしきい値は0.5固定です。
- 遮蔽率でGTを取り除きません。全GTのRecallと、visibilityの高・中・低別のRecallを出します。
  高は0.7以上、中は0.3以上0.7未満、低は0.3未満です。
- 原論文で1始まりとされる座標を0始まりへ変換し、矩形を画像内へ切り詰めます。
  元の矩形・クラス・active・visibilityも残します。
- 認識評価は、同クラスの矩形IoU≥0.5の一対一割当てです。
  対応数を最大化した後、IoU総和を最大化します。小さな行列の全探索と比較して検証しました。
- 生のモデル個数は除外ルールで書き換えません。そのため、個数差をactive歩行者数と
  比較する指標には対象定義の差が含まれます。現場の物理的変化の誤警報率として使えません。

これはDeltaSenseの2画像比較用の診断であり、MOTChallengeの公式評価手順そのものではありません。
MOTA、IDF1、HOTAなどの公式追跡指標やランキングを算出・主張しません。
マスク・物理単位の正解はなく、それらの精度も評価しません。
元ラベルはデータセット作者の人手ラベルですが、今回の変換を独立した人が確認した状態では
ありません。manifest内の `independent_human_review=false` は、その確認状況を表します。

## 通信を遮断した実行

素材・依存関係・モデルの準備後に実行します。出力先は未作成のディレクトリを指定します。

```sh
YOLO_CONFIG_DIR="$PWD/artifacts/yolo-config" \
  unshare --user --map-root-user --net .venv/bin/python scripts/evaluate_accuracy.py \
  --manifest artifacts/mot17/manifest-v1.json --assets artifacts/mot17 \
  --models artifacts/models --output artifacts/mot17/new-baseline \
  --require-isolated-network --plot-every 8 --review-every 16
```

同じラベルとモデルで行った探索的な設定比較は `mot17-experiments-v1.json` に記録しています。
同じコマンドに `--imgsz 1280`、または `--imgsz 1280 --conf 0.1` を指定し、
それぞれ別の出力先へ実行しました。両側の画像で同じ設定を使います。
しきい値比較は同じ評価画像の再利用なので、未使用データでの最終評価ではありません。
この結果だけでライブラリの既定値を変更していません。

## 失敗の内訳と対応しきい値の比較

元の推論結果を再利用して、認識不足・候補競合・別IDとの対応・候補なしを集計します。

```sh
.venv/bin/python scripts/analyze_mot17.py \
  --manifest artifacts/mot17/manifest-v1.json \
  --runs artifacts/mot17/run-v1 artifacts/mot17/run-1280-v1 \
    artifacts/mot17/run-1280-conf010-v1 \
  --output artifacts/mot17/new-attribution.json

.venv/bin/python scripts/evaluate_matching.py \
  --manifest artifacts/mot17/manifest-v1.json \
  --runs artifacts/mot17/run-v1 artifacts/mot17/run-1280-v1 \
    artifacts/mot17/run-1280-conf010-v1 \
  --output artifacts/mot17/new-matching-sweep.json
```

対応しきい値の比較ではJSONの矩形から候補関係だけを再構成します。
元のmatch_iou=0.2で、対応・曖昧候補・未対応・個数が元結果と一致することを先に検証します。
セグメンテーションのマスクは再構成せず、その精度も比較しません。
使用した予測ファイルのSHA-256を記録し、変更されていないことを確認します。

結果は [日本語レポート](../docs/mot17-accuracy.md) にあります。
