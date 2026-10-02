# Project concept

Most computer vision libraries analyze one image at a time. DeltaSense is about the gap between two observations.

## プロジェクトのゴール（センターピン）

> 同じ対象の対応する領域を異なる時点で比較し、撮影条件による見え方の違いを考慮しながら、対象にとって意味のある変化を、その内容・量・根拠とともに説明する。測定した事実と解釈を区別し、判断できない場合は理由を示す。

利用者が知りたいのは、「何が、どのくらい変わったのか」「そう判断できる根拠は何か」です。
対象の対応付け、比較できる撮影条件の確認、変化の測定、その意味の説明を、このゴールに向けて進めます。
量を測れない場合は、測れない理由を示し、数値を作りません。

### 測定と解釈の具体例

以下は目指す利用例であり、v0.1で実現済みの機能ではありません。

| 利用者の問い | 画像から測定・観察するもの | 解釈に必要な根拠・限界 |
| --- | --- | --- |
| 同じ道にいる人は前に進んだ？ | 対応する人物の位置の差 | 同じ人物であること、道路に対する位置、撮影位置の違いを確認する。「前」が道路の進行方向・本人の向き・カメラ側のどれなのかも定義する。 |
| 1年前より髪が薄くなった？ | 対応する頭部領域の髪や頭皮の見え方の差 | 光、角度、髪型などによる違いを考慮する。頭皮が多く見えるだけでは、髪が減ったとは断定しない。 |
| 顔を比べて太ったか知りたい | 対応する顔領域の輪郭や頬の見え方の差 | 距離、角度、表情などによる違いを考慮する。顔の見え方だけで体重の増加を確定することはできない。 |

例えば人物の画像座標が変わったことは測定結果です。
そこから「道路の進行方向へ移動した」と説明するには、道路を基準にした比較と、それを支える根拠が必要です。
2枚の写真から分かるのは撮影時点の位置の違いであり、途中の経路や歩き続けていたことまでは分かりません。

### 進む方向を判断する基準

- 同じ対象の対応する領域を比較し、比較対象や方向、単位を明示できる。
- 撮影条件だけが変わった例で、対象の状態が変わったと誤って説明しない。
- 対象が実際に変わった例で、変化の内容・量・根拠を示せる。
- 根拠が足りない例で、判断できない内容と理由を示せる。

検証では、用途ごとの正解付きbefore／afterで見逃し・誤警報・対応付けを評価します。
判断を保留した割合も併せて確認し、すべてを「判断不能」にするだけでは達成としません。
頭部や顔は代表例です。特定の部位専用の製品に限定することは、このゴールに含めません。

## ゴールと現在のバージョンの関係

v0.1は、モデルが観測した結果を比較し、測定結果と限界を返すための基盤です。
撮影条件を自動で確認したり、「人が前進した」「髪が薄くなった」「体重が増えた」と解釈したりする機能は未実装です。

予定されているv0.2の限定的なカメラ位置合わせと幾何学的な妥当性確認は、比較可能性を改善する一歩です。
それだけで、照明・姿勢・奥行きの違いや、用途ごとの意味の解釈がすべて解決するわけではありません。
このゴールはプロジェクト全体の方向性であり、既存のv0.1の完了条件や、後続バージョンの実装済み機能を表すものではありません。

## Initial implementation approach

The project starts with Ultralytics because it already gives detection, segmentation, classification, pose, oriented boxes, and depth a consistent result format. DeltaSense can run one selected model on two images and compare those results.

## Initial direction

v0.1 implements detection and instance-segmentation
comparison. The original starting direction below describes the product intent;
[release requirements](requirements.md) and [the API contract](api.md) define the
implemented scope. See [validation](validation.md) for what has actually been tested.

The first useful version should stay small:

1. Accept a before image and an after image.
2. Run the same Ultralytics model on both.
3. Compare the returned results.
4. Report changes in a plain Python object that can also be written as JSON.

v0.1 compares prediction counts, conservative spatial correspondences, bbox
displacement/area, and mask area/support differences. It does not infer physical
shape changes from a spatial mask difference. Pose and depth should wait until a
real use case needs them.

## What model output can and cannot prove

A difference between predictions is not always a difference in the photographed subject. The camera may have moved. Lighting may have changed. An object may be hidden, blurred, or missed by the model.

DeltaSense reports those limits instead of turning every difference into a change
claim. v0.1 distinguishes measurement availability and explicitly unchecked capture
conditions. It does not assign a global verified-comparable/unobservable/confounded
verdict without the evidence to support it.

## Why keep it generic

The same comparison pattern appears in many places: a scratch on a manufactured part, growth in a plant, work completed on a building, or a subtle change in a person. The models and measurements differ, but the before and after workflow repeats.

DeltaSense should not include domain logic until someone brings a real example and a test. The initial project only needs to prove that one small interface can compare Ultralytics results without hiding what the model actually observed.
