

# **DeBERTaを用いた二値分類における不変リスク最小化（IRM）導入のための複数トレーニング環境構築とtransformersフレームワークでの実装に関する技術報告書**

## **I. IRM（不変リスク最小化）の理論的基盤とNLPにおける必要性**

### **A. 機械学習における汎化の限界と分布シフトの問題**

従来の深層学習モデルは、経験的リスク最小化（Empirical Risk Minimization, ERM）パラダイムに基づいて訓練されます。ERMは、訓練データとテストデータが同一の独立同分布（i.i.d.）から抽出されるという統計的な前提が成立する場合に、優れた性能を発揮します 1。しかし、現実世界、特に自然言語処理（NLP）の分野では、データの収集源、時間経過、トピックの変化などにより、訓練時と配備時でデータの分布が変化する、いわゆる「分布シフト（Distribution Shift）」が頻繁に発生します 1。

この分布シフトには、特徴量（X）の分布のみが変わる共変量シフトや、特徴量とラベル（Y）の関係性自体が変わる概念シフトなどが含まれます 1。モデルがこれらのシフトに直面すると、訓練データ上では高精度であっても、未見のデータセット（Out-of-Distribution, OOD）に対する性能は著しく低下する可能性があります 1。

### **B. スパリアス相関（Shortcut Learning）の定義とDeBERTaモデルへの影響**

分布シフトへの脆弱性の主要因は、モデルが目的タスクの真の因果構造ではなく、特定の訓練環境でのみ存在する統計的な依存関係、すなわち「スパリアス相関（Spurious Correlations）」を学習してしまうことにあります 4。これは「ショートカット学習」とも呼ばれ、モデルが最も簡単な経路で損失を最小化しようとする結果として生じます。

DeBERTaのような強力な事前学習済み言語モデル（PLM）は、RoBERTaやBERTを凌駕する高い表現能力を持ち、特に分散型アテンション（disentangled attention）などの技術により、効率的にパターンの学習を行います 6。この非常に高いパターン認識能力は、訓練データセットにわずかでもスパリアス相関が存在すれば、それを**非常に効率的に**学習し、予測に利用してしまう傾向を強化します 4。

NLPにおける具体的な例として、毒性判定のような二値分類タスクにおいて、特定のデータソース（例：Twitter）からのデータがネガティブなラベルに偏っている場合、モデルはテキストの内容が真に毒性を持つかどうかに関わらず、そのテキストの「ソース」や、ソース特有の文体、あるいは許容されるテキスト長の特性 7 をショートカットとして利用し、高性能を達成することがあります。しかし、ソースが異なる新しいデータセットに適用された場合、このモデルは完全に機能不全に陥ります。

### **C. IRM原理の定式化：不変な特徴量表現の学習**

不変リスク最小化（IRM）は、このようなスパリアス相関の利用を回避し、異なる環境を跨いで一貫して予測性能を持つ、**不変な因果的特徴**を抽出するための学習パラダイムとして提案されました 8。

IRMの核心的な目的は、データ表現関数 $\\phi: X \\to H$ を学習することであり、この表現の上に乗る最適な線形分類器 $w: H \\to Y$ が、全ての訓練環境の集合 $E\_{tr}$ において一貫して同じである、という制約を満たすようにします 8。

IRMは、学習データを単に分割するのではなく、真の因果的要因とスパリアスな統計的要因の間に存在する**分布シフトを意図的に露出させる**ことで、モデルがどの特徴が環境によらず不変であるかを識別できるように強制します。標準的なランダム分割では、i.i.d.の前提を維持しようとするため、この重要な因果的な分布シフトが均されてしまい、IRMの目的である不変な特徴学習を効果的に行うことができません。

### **D. IRMv1の目的関数とバイレベル最適化の概念**

IRM原理は、元来バイレベル最適化問題として定式化されます 11。

$$\\min\_{\\phi: \\mathcal{X} \\to \\mathcal{H}} \\sum\_{e \\in E\_{tr}} R\_e(\\phi) \\quad \\text{subject to} \\quad w \\in \\arg\\min\_{\\tilde{w}: \\mathcal{H} \\to \\mathcal{Y}} R\_e(\\tilde{w} \\circ \\phi), \\quad \\forall e \\in E\_{tr}$$  
ここで $R\_e(\\phi)$ は環境 $e$ における経験的リスク（損失）です。この厳密な定式化は計算上困難であるため、実用的な実装としてIRMv1が提案されました 11。IRMv1は、バイレベルの制約を、線形分類器 $w$ に関する損失勾配のノルムをペナルティ項として課す形に緩和します。

IRMv1の目的関数 $L\_{IRMv1}$ は、すべての訓練環境 $E\_{tr}$ における経験的リスクの合計と、不変性ペナルティ項 $\\Omega$ の加重和として表現されます。

$$L\_{IRMv1} \= \\sum\_{e \\in E\_{tr}} R\_e(\\phi, w) \+ \\lambda \\cdot \\Omega$$  
ここで $\\Omega$ は勾配ペナルティ項であり、以下のように定義されます（ここでは分類器 $w$ をダミースカラーに固定した上での勾配ノルムの二乗の合計として扱う）。

$$\\Omega \= \\sum\_{e \\in E\_{tr}} \\| \\nabla\_w R\_e(w \\circ \\phi)|\_{w=1} \\|^2$$  
この目的関数を最小化することで、モデルは全ての環境で経験的リスクを最小化するだけでなく、環境間で分類器の重みが変化しないように（すなわち、特徴量 $\\phi(X)$ が環境不変であるように）強制されます。

## **II. DeBERTa/二値分類のためのトレーニング環境構築戦略**

IRMを成功させるための最も重要な前提は、訓練データセット $D\_{train}$ を、分布シフトを意図的に露出させた**複数の離散的な環境** $E\_e$ に分割することです 12。単純なランダム分割は、データ全体からi.i.d.のサンプルを取り出すため、分布シフトを意図的に取り込むというIRMの要件を満たしません。

環境の構築は、スパリアス相関がラベルとの関係で変動するように設計されなければなりません。以下に、DeBERTaを用いた二値分類タスクで適用可能な3つの主要な環境構築戦略を提示します。

### **A. 戦略 1: 明示的なメタデータに基づく分割**

データセットに付随する構造化情報、すなわちメタデータは、自然な環境ラベルとして活用できる可能性が高いです 7。メタデータは、テキストそのものの特徴（例：長さ、感情）とは異なり、データの背景情報（例：ソース、作成日時、著者属性）を提供します。

1. データのソース/ドメインによる分割:  
   テキストデータが複数の異なるプラットフォーム（例：ニュース記事、ソーシャルメディア、フォーラム）から収集されている場合、それぞれのソースを独立した環境 Ee​ とします 7。ソースが異なれば、データの分布（語彙、文法、文脈）が自然に異なり、共変量シフトが導入されます。もし、あるソースに特有のバイアス（例：特定のプラットフォームのユーザーは特定の意見に偏る）が存在すれば、それがスパリアス相関として機能し、IRMのペナルティのターゲットとなります。  
2. 作成日時/時間による分割（Temporal Split）:  
   データの作成時間や収集時期を環境ラベルとして使用する戦略は、時間経過に伴う自然な分布シフト（Temporal Shift）に対応します 2。例えば、政治的なトピックや健康問題（例：COVID-19関連データ）に関する分類タスクでは、数ヶ月や数年で議論の焦点や用語が変化します。このような概念シフトを環境として分離することで、モデルは時間不変な特徴を学習するように促されます。  
3. 特定の著者属性に基づく分割:  
   データ収集時に利用可能な著者やユーザーの属性（例：年齢、地域、デモグラフィック情報）を環境として使用します。これにより、特定の属性に関連するバイアス（例：ある地域特有の表現）がスパリアス属性として露出します 7。

### **B. 戦略 2: スパリアス属性の特定と人為的なバイアス導入による分割**

環境ラベルが利用できない、またはバイアス源が明確なメタデータとして存在しない場合、モデルがショートカットとして利用する可能性のある「スパリアス属性」を事前に特定し、それに基づいて訓練データを人為的に構築することができます 13。

このアプローチは、モデルに特定の相関関係を学習させないように強制する因果的学習手法（例：GroupDROやSCIILの概念）から着想を得ています 5。

1. **環境の設計:** 二値分類タスク（$Y \\in \\{0, 1\\}$）と、スパリアス属性 $A$ を定義します。  
2. **環境 $E\_1$（相関が強い環境）:** ラベル $Y$ と属性 $A$ が強く相関するようにデータをサンプリングまたは操作します（例：毒性ラベルY=1のサンプルは、特定のネガティブなキーワードAを多く含む）。  
3. **環境 $E\_2$（相関が弱いまたは反転した環境）:** ラベル $Y$ と属性 $A$ の相関が $E\_1$ とは大きく異なるようにデータをサンプリングします。理想的には、属性 $A$ が存在するにもかかわらず、ラベル $Y$ が反対であるサンプル（例：ネガティブなキーワードAを含むが、実際は非毒性Y=0のサンプル）を意図的に多く含めます。

IRMは、この相関の変動を利用して、予測性能を安定させるために属性 $A$ を無視し、環境間で不変な真の因果的特徴に依存するように学習します 5。

### **C. 戦略 3: 敵対的検証（Adversarial Validation, AV）を用いた環境の自動推定**

環境ラベルが存在せず、かつスパリアス属性の特定が困難な場合に、敵対的検証（AV）は、データセット内の潜在的なドメインシフトや分布の不一致を自動的に特定し、IRMに必要な環境を生成する強力な手段となります 3。

#### **1\. AVのメカニズム**

AVは、データセット $D\_{train}$ の特徴量 $X$ の分布のみに基づいて、データポイントがどのサブセットに属するかを予測する二値分類タスクです 16。

1. **初期分割:** 訓練データ $D\_{train}$ をランダムに2つのグループ $D\_{A}$ と $D\_{B}$ に分割します。  
2. **ドメイン分類器の訓練:** $D\_{A}$ のサンプルにラベル 0、$D\_{B}$ のサンプルにラベル 1を割り当て、DeBERTaのテキスト特徴量（またはその他の特徴量）を入力として、このドメインラベルを予測する分類器（通常はLightGBMやXGBoostなどの非深層学習モデル）を訓練します 17。  
3. **分布シフトの測定:** このドメイン分類器のROC AUCスコアが評価されます 16。  
   * AUCが 0.5 に近い場合：$D\_{A}$ と $D\_{B}$ の分布は非常に類似しており、分布シフトは弱い。  
   * AUCが 0.7 などの高い値を示す場合：ドメイン分類器は $D\_{A}$ と $D\_{B}$ を容易に区別できる、すなわち、強い分布シフトが存在する 17。

#### **2\. 環境ラベルの生成**

訓練されたAVモデルを $D\_{train}$ 全体に適用し、各サンプルがグループ 1 に属する確率 $P\_{AV}(1|X)$ を予測します。この $P\_{AV}$ スコアは、そのサンプルが訓練セット全体の主要な分布からどれだけ「異なっている」かを示す指標となります 3。

IRMで要求される離散的な環境ラベル $E$ を生成するためには、この連続的な $P\_{AV}$ スコアを二値化する必要があります 18。

* **閾値設定:** $P\_{AV}$ スコアに対して閾値 $\\tau$（例：0.5、または特定のパーセンタイル値）を設定します。  
* $P\_{AV}(1|X) \\ge \\tau$ のサンプルを環境 $E\_1$（シフトが強いグループ）とし、残りを環境 $E\_2$（シフトが弱いグループ）と定義します。

この手法の利点は、人間が認識できない特徴量に基づく未知の分布シフトを客観的に数値化し、その分布的な分離度合いに基づいてIRMに必要な環境を自動で生成できる点にあります 3。これにより、モデルは分布的に異なる環境間でロバストな特徴を学習するように強制されます。実務上、安定した学習のためには、強い分布シフトを内在する少数のサンプル（高い $P\_{AV}$ スコアを持つ）を $E\_1$ として隔離し、残りを $E\_2$ とする極値閾値化戦略も有効である可能性があります 20。

Table 1: IRMのための環境構築戦略比較

| 戦略名 | 環境定義の基準 | データセット要件 | IRMへの効果 | 実務上の複雑性 |
| :---- | :---- | :---- | :---- | :---- |
| メタデータ分割 | 既知の外部属性（ソース、日時、著者属性） 7 | 信頼性の高い環境ラベルが必要 12 | 既知のドメインシフト（共変量/コンセプト）を直接分離 | 低〜中 |
| スパリアス属性分割 | 意図的に設計された、ラベルと相関する見せかけの属性 13 | 属性の特定とラベリング、または人為的なラベル/特徴量操作が必要 | スパリアス相関の排除に最も効果的（因果的バイアスを直接標的） | 高 |
| 敵対的検証 (AV) | 訓練データ内で分布が最も異なるサブセットの自動検出 3 | 環境ラベル不要。高いROC AUCを持つ特徴量に基づく 17 | 未知の分布シフトを検出し、その強さに応じてデータ分割が可能 16 | 中 |

## **III. Hugging Face transformersを用いたIRMの実装パターン**

DeBERTaモデルをtransformersフレームワーク内でIRMv1の原則に従って訓練するためには、標準的な訓練ループでは不十分であり、カスタムの損失計算ロジックを導入する必要があります。これは、特にIRMv1の勾配ペナルティの計算に、PyTorchの高度な自動微分機能が必要となるためです。

### **A. transformers.Trainerクラスのカスタム化**

Hugging Faceのエコシステムでは、訓練のカスタマイズは主にクラスの継承と、その内部メソッドであるcompute\_lossのオーバーライドによって行われます 21。

標準のTrainerは、入力バッチを受け取り、モデルのフォワードパスを実行し、単一の損失値（二値分類であれば通常はCross-Entropy Loss）を返すように設計されています 23。しかし、IRMでは、バッチを環境ごとに分割し、各環境の損失 $R\_e$ と勾配ペナルティ $\\Omega$ を計算し、それらを統合した $L\_{IRMv1}$ を返す必要があります 11。

### **B. 環境グループのデータローダー設計**

IRMv1の目的関数を計算するためには、単一のミニバッチ内に、複数の異なる訓練環境からのサンプルが混在している必要があります。また、各サンプルがどの環境に属するかを識別するための環境ID（$e$）が必要です 12。

推奨されるデータローダーのパターンは以下の通りです。

1. **環境ごとのデータセット:** 事前に定義された環境 $E\_1, E\_2, \\dots$ ごとに独立した オブジェクトを作成します。  
2. **バッチ構造:** データローダーは、入力データ（input\_ids, attention\_mask）、ターゲットラベル（labels）に加え、環境ID（例：environment\_id）を含むように設計されます。  
3. **サンプリング:** 最もシンプルな方法は、各環境のデータセットからランダムにサンプリングした少数のサンプルを連結し、単一のIRMバッチを構成することです。これにより、compute\_lossメソッド内で環境ごとの分離が可能になります。

### **C. DeBERTaの出力の活用とモデルの論理的分割**

IRMv1の定式化は、モデルを以下の2つの論理的な構成要素に明確に分割することを前提としています 11。

1. **特徴量抽出器 ($\\phi$)**: 入力 $X$（トークンID、アテンションマスクなど）を受け取り、不変な特徴表現 $H$ を生成する部分。DeBERTaモデル全体（埋め込み層から最終Transformer層まで）がこれに該当します。具体的には、最終層の分類トークン（\`\`トークン）の隠れ状態出力が表現 $H$ と見なされます 6。  
2. **線形分類器 ($w$)**: 特徴量 $H$ を受け取り、最終的なロジット $Y$ を出力する部分。通常、モデルの最後の線形層（classifierまたは同様の名前の層）がこれに該当します。

IRMv1のペナルティ項 $\\Omega$ は、この**線形分類器 $w$ の重み**に対してのみ、損失の勾配の不変性を強制します。

### **D. IRMv1損失の計算フロー（PyTorch/Autogradの利用）**

カスタムcompute\_loss内でのIRMv1損失の計算手順は以下の通りです。

1. 特徴量の抽出と環境への分離:  
   入力バッチを受け取り、DeBERTaのフォワードパスを実行して、分類層に入力される前の特徴量 H（\`\`トークンの表現）を取得します。次に、環境ID（environment\_id）に基づいて、特徴量 H とラベル Y を環境 E1​,E2​,… ごとに分離します。  
2. 線形分類器 w の設定:  
   IRMv1の標準的な実装では、勾配ペナルティを計算する際に、分類器 w の重みをダミーのスカラー値（通常は 1.0）に設定します。これは、元のIRMの制約を近似するための簡略化です。  
3. 環境ごとの経験的リスク Re​ の計算:  
   各環境 e について、分離された特徴量 He​ とラベル Ye​ を使用して、線形分類器 w を通した予測ロジットを計算し、標準的な二値分類損失（例：torch.nn.functional.cross\_entropy）Re​ を計算します 23。

   $$\\text{Total ERM Loss} \= \\sum\_{e \\in E\_{tr}} R\_e$$  
4. 不変性ペナルティ Ω の計算 (勾配ペナルティ)  
   ここが最も技術的に要求される部分であり、PyTorchのtorch.autograd.grad機能を利用します 26。  
   * **勾配の取得:** 各環境 $e$ の損失 $R\_e$ について、**分類器 $w$ の重み**に関する勾配 $\\nabla\_w R\_e$ を計算します。  
     Python  
     \# R\_e は環境eの損失  
     \# w\_params は分類器 w のパラメータ  
     grad\_params \= torch.autograd.grad(R\_e, w\_params, create\_graph=True)

     ここで create\_graph=True は、計算された勾配自体（grad\_params）を、その後のペナルティ計算のために微分可能な非リーフ変数として保持するために不可欠です 26。  
   * ノルムの計算: 各環境で得られた勾配ベクトルのノルム二乗を計算します。

     $$\\Omega \= \\sum\_{e \\in E\_{tr}} \\left( \\sum\_{p \\in w} \\| \\nabla\_{p} R\_e \\|^2 \\right)$$  
5. 最終的な LIRMv1​ の構成:

   $$\\mathbf{L\_{IRMv1}} \= \\sum\_{e \\in E\_{tr}} R\_e \+ \\lambda \\cdot \\Omega$$

   ここで λ はハイパーパラメータであるペナルティ重みです。このスカラー損失値をcompute\_lossから返し、Trainerが一度のバックプロパゲーション（.backward()）を実行します。これにより、特徴量抽出器 ϕ と分類器 w の両方が、ペナルティを考慮した勾配で同時に更新されます。

Table 2: IRMv1損失関数の構成要素とPyTorch実装の対応

| 要素 | 数学的表現 | 役割 | DeBERTa/PyTorch実装の対応 |
| :---- | :---- | :---- | :---- |
| 特徴量抽出器 | $\\phi$ | DeBERTaエンコーダ（Transformer層） | model.deberta |
| 線形分類器 | $w$ | $H \\to Y$ にマップするヘッド | model.classifier |
| 経験的リスク (ERM) | $R\_e(\\phi, w)$ | 標準的な分類誤差の最小化 | F.cross\_entropy (各環境で計算) 23 |
| 勾配ペナルティ項 | $\\|\\nabla\_w R\_e(\\dots)\\|$ | 特徴量 $\\phi$ の不変性の強制 11 | torch.autograd.gradによる $w$ に関する勾配ノルム計算 26 |
| ペナルティ重み | $\\lambda$ | 不変性制約の強さ | ハイパーパラメータ（スケジューリング推奨）27 |

## **IV. IRMv1最適化と勾配ペナルティの技術的詳細**

IRMv1の最適化は、標準的なERMと比較して著しく不安定になることが知られています 27。これは、特徴量抽出器 $\\phi$ を更新するための勾配が、ERM項だけでなく、勾配ペナルティ項 $\\Omega$ を通じても流れるためです。この複雑な結合勾配を管理することが、実用的な実装の鍵となります。

### **A. 安定性の確保：フィーチャ抽出器 $\\phi$ の縮退問題**

IRMv1の学習が失敗する最も一般的なケースは、ペナルティ重み $\\lambda$ が高すぎることにより、特徴量抽出器 $\\phi$ が\*\*縮退解（Degenerate Solution）\*\*に収束してしまうことです 27。

縮退とは、$\\phi(X)$ の出力が全ての入力 $X$ に対して単一の定数値に近づく状態を指します。この状態になると、表現空間 $H$ 上では、どの線形分類器 $w$ も等しく最適（または非最適）となり、結果として任意の $w$ に対する損失 $R\_e$ の勾配 $\\nabla\_w R\_e$ はゼロに近づきます。ペナルティ $\\Omega$ は最小化されますが、モデルは分類能力を完全に失い、二値分類では精度が50%（ランダム推測レベル）に固定されてしまいます 27。

### **B. ペナルティ重み ($\\lambda$) のスケジューリング戦略**

この不安定性を回避し、かつ不変性を効果的に強制するためには、ペナルティ重み $\\lambda$ の慎重な管理が不可欠です 27。

1. 初期 ERMフェーズ (λ=0):  
   訓練の初期段階では、λ をゼロに設定し、IRMを通常のERMとして実行します。これにより、DeBERTaモデルはまずデータセットから基本的な特徴量表現を学習します。この段階では、モデルがスパリアス相関を利用して高い訓練精度を達成することが許容されます 12。  
2. 線形増加（ペナルティの導入）:  
   初期のERMフェーズが完了した後、または特定のエポックに達した後、λ をゼロから目標値へと徐々に増加させます。この漸進的な増加により、モデルは安定した特徴量表現を維持しつつ、不変性の制約を満たす方向にゆっくりと誘導されます 27。これは、モデルが既に学習したスパリアスな依存関係を識別し、それを打ち消すように ϕ を調整する時間を与えます。  
3. 早期終了基準:  
   IRMの訓練は、訓練環境におけるERM損失やペナルティ項の変動だけでは収束を判断するのが難しいため、独立した検証環境 Eval​ を設定し、そのOOD性能に基づいて早期終了（Early Stopping）を適用することが強く推奨されます 27。これにより、モデルが不変性制約を過剰に満たそうとして縮退解に向かう前に、最適な汎化性能を持つポイントで訓練を停止できます。このため、実用上は少なくとも3つ以上の環境（訓練環境 E1​,E2​ および検証環境 Eval​）が必要となります 27。

### **C. モデルパラメータの管理と勾配ペナルティの適用**

勾配ペナルティを正しく適用するためには、DeBERTaモデルのパラメータを論理的に分離し、正確にターゲット指定する必要があります。

* **特徴量抽出器 $\\phi$ のパラメータ:** DeBERTaのエンコーダ層全体の重み。  
* **線形分類器 $w$ のパラメータ:** 分類ヘッドの重み。

カスタムcompute\_loss関数内で、線形分類器 $w$ の重みを明示的に取得し、その重みを torch.autograd.grad のターゲットとして渡す必要があります。transformersのモデル構造（例：model.classifier.weight, model.classifier.bias）を深く理解し、これらのパラメータに対する勾配のみを計算し、そのノルムを損失に加算する処理を実装しなければなりません 26。

この分離と特定のパラメータをターゲットとする勾配計算の正確さが、IRMv1実装の成功を決定します。

## **V. IRMの評価と実運用における注意点**

### **A. IRMモデルの評価指標：OODデータセットでの性能測定**

IRMの目的は、訓練データセットの分布から外れた（OOD）データに対してロバストに機能することであるため、評価は必ず独立したテスト環境 $E\_{test}$ で行われるべきです 10。

1. 独立したOODテスト環境:  
   Etest​ は、訓練環境 Etr​ とは異なるソース、異なるバイアス特性、または異なる時点から収集されたデータである必要があります。この Etest​ の性能が、モデルの真の汎化能力を示します。  
2. ワーストケースグループの性能:  
   単なる平均精度だけでなく、構築された訓練環境、または既知のバイアスグループにおける最も性能が低いサブグループの性能を測定することが重要です 3。GroupDROなど、分布的にロバストな最適化手法は、特にこのワーストケース性能の改善を目指します。二値分類におけるAUC、精度、F1スコアが主要な評価指標となりますが、OOD環境でこれらの指標が安定していることが、不変特徴量が適切に学習されたことの証拠となります 28。

### **B. IRMの限界と代替手法**

IRMは強力なフレームワークですが、その実装と運用にはいくつかの困難が伴います。

* **環境ラベルの必要性:** IRMは、訓練データが複数の環境に分割され、その環境ラベルが既知であることを前提とします 12。環境ラベルの定義が不適切であったり、ラベル自体が入手不可能であったりする場合、IRMは効果を発揮しません。  
* **最適化の不安定性:** 前述の通り、$\\lambda$ の選択やスケジューリングが訓練の安定性に致命的な影響を与えます 27。

環境ラベルがない、または不安定性が懸念される場合の代替手法として、以下のものが考慮されます。

1. Group Distributionally Robust Optimization (GroupDRO):  
   これは、全訓練環境の中で最も損失が大きい（すなわち、最も性能が悪い）環境の損失を最小化することに焦点を当てる手法です。IRMと比較して実装が容易であり、特に環境の事前分割が可能な場合に有効なロバスト化戦略です 2。  
2. Environment Inference for Invariant Learning (EIIL):  
   EIILは、環境ラベルが全く与えられていない状況に対応するために提案されたフレームワークです 3。これは、敵対的な手法を用いて、不変学習に最も有用な環境の分割（環境ラベル）をモデルの訓練と並行して推定します 12。EIILは、最初にスパリアス相関に強く依存するERMモデルで初期化される必要があり、その後の環境推論によって、未知の分布シフトに対処します 12。

### **C. 結論と推奨事項**

DeBERTaを用いた二値分類タスクにIRMを導入するための「複数の異なるトレーニング環境」を構築するという課題は、単純なデータ分割では達成できません。

本報告書で詳述された分析に基づき、以下の結論と具体的な推奨事項が導かれます。

#### **1\. 環境構築に関する結論：分布シフトの意図的な最大化**

IRMの成功は、単にデータを2つ以上に分割することではなく、**各環境における特徴量とラベルの同時分布 $P\_e(X, Y)$ が互いに異なり、特にスパリアス相関の性質が環境間で変動するように**、環境を設計することにかかっています。

* **推奨される分割方法:** 環境ラベルが存在する場合（メタデータ、スパリアス属性）は、それらを活用した戦略（II. A, II. B）。環境ラベルがない、または未知の分布シフトが疑われる場合は、敵対的検証（AV）を用いた自動推定戦略（II. C）が最もロバストな環境分割を提供します 3。  
* **環境の最小数:** 安定した訓練とハイパーパラメータの調整のために、少なくとも3つ以上の独立した環境（訓練 $E\_1, E\_2$ および OOD 検証 $E\_{val}$）を用意することが不可欠です 27。

#### **2\. 実装に関する結論：カスタム損失の厳密な定義**

transformersフレームワークでの実装では、カスタムcompute\_lossメソッドをオーバーライドし、IRMv1の目的関数を正確に計算する必要があります。

* **勾配ペナルティの厳密性:** DeBERTaの分類ヘッド $w$ の重みに対してのみ、torch.autograd.grad(..., create\_graph=True) を使用して損失勾配のノルム $\\Omega$ を計算することが、不変性制約を正確に適用するための技術的な要件です 11。  
* **最適化の安定性:** ペナルティ重み $\\lambda$ は、訓練の初期段階ではゼロから開始し、モデルが縮退解に陥るのを防ぐために、訓練の進行に伴って徐々に増加させるスケジューリング戦略を適用することが、実用的な安定性の鍵となります 27。

#### **引用文献**

1. Handling Out-of-Distribution Data: A Survey \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/html/2507.21160v1](https://arxiv.org/html/2507.21160v1)  
2. A Benchmark of in-the-Wild Distribution Shift over Time \- OpenReview, 10月 21, 2025にアクセス、 [https://openreview.net/pdf?id=BUQD1tJ2UwK](https://openreview.net/pdf?id=BUQD1tJ2UwK)  
3. Environment Inference for Invariant Learning \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/pdf/2010.07249](https://arxiv.org/pdf/2010.07249)  
4. Invariant Language Modeling \- Microsoft Research, 10月 21, 2025にアクセス、 [https://www.microsoft.com/en-us/research/publication/invariant-language-modeling/](https://www.microsoft.com/en-us/research/publication/invariant-language-modeling/)  
5. Spurious Correlations in Machine Learning: A Survey \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/pdf/2402.12715](https://arxiv.org/pdf/2402.12715)  
6. DeBERTa \- Hugging Face, 10月 21, 2025にアクセス、 [https://huggingface.co/docs/transformers/en/model\_doc/deberta](https://huggingface.co/docs/transformers/en/model_doc/deberta)  
7. NLP Metadata — Deepchecks Documentation, 10月 21, 2025にアクセス、 [https://docs.deepchecks.com/stable/nlp/usage\_guides/nlp\_metadata.html](https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_metadata.html)  
8. \[1907.02893\] Invariant Risk Minimization \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/abs/1907.02893](https://arxiv.org/abs/1907.02893)  
9. \[2405.01389\] Invariant Risk Minimization Is A Total Variation Model \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/abs/2405.01389](https://arxiv.org/abs/2405.01389)  
10. Invariant Risk Minimization: Learning Explanations That Generalize Across Environments, 10月 21, 2025にアクセス、 [https://eureka.patsnap.com/article/invariant-risk-minimization-learning-explanations-that-generalize-across-environments](https://eureka.patsnap.com/article/invariant-risk-minimization-learning-explanations-that-generalize-across-environments)  
11. On Invariance Penalties for Risk Minimization \- White Rose Research Online, 10月 21, 2025にアクセス、 [https://eprints.whiterose.ac.uk/id/eprint/224734/1/2106.09777v1.pdf](https://eprints.whiterose.ac.uk/id/eprint/224734/1/2106.09777v1.pdf)  
12. Environment Diversification with Multi-head Neural Network for Invariant Learning, 10月 21, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2022/file/062d711fb777322e2152435459e6e9d9-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/062d711fb777322e2152435459e6e9d9-Paper-Conference.pdf)  
13. Navigating Shortcuts, Spurious Correlations, and Confounders: From Origins via Detection to Mitigation \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/html/2412.05152v1](https://arxiv.org/html/2412.05152v1)  
14. Robustness to Spurious Correlation: A Comprehensive Review \- OOD-CV, 10月 21, 2025にアクセス、 [https://www.ood-cv.org/camera\_ready/W47.29.pdf](https://www.ood-cv.org/camera_ready/W47.29.pdf)  
15. Adversarial Validation Made Simple \- Kaggle, 10月 21, 2025にアクセス、 [https://www.kaggle.com/code/lusfernandotorres/adversarial-validation-made-simple](https://www.kaggle.com/code/lusfernandotorres/adversarial-validation-made-simple)  
16. What is Adversarial Validation? \- Kaggle, 10月 21, 2025にアクセス、 [https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation](https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation)  
17. Automatic adversarial validation \- Kaggle, 10月 21, 2025にアクセス、 [https://www.kaggle.com/code/somnambwl/automatic-adversarial-validation](https://www.kaggle.com/code/somnambwl/automatic-adversarial-validation)  
18. AdvDINO: Domain-Adversarial Self-Supervised Representation Learning for Spatial Proteomics \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/html/2508.04955v1](https://arxiv.org/html/2508.04955v1)  
19. Binarizing Split Learning for Data Privacy Enhancement and Computation Reduction \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/pdf/2206.04864](https://arxiv.org/pdf/2206.04864)  
20. Adversarial Validation to Select Validation Data for Evaluating Performance in E‑commerce Purchase Intent Prediction \- SIGIR eCom, 10月 21, 2025にアクセス、 [https://sigir-ecom.github.io/ecom21DCPapers/paper3.pdf](https://sigir-ecom.github.io/ecom21DCPapers/paper3.pdf)  
21. Trainer \- Hugging Face, 10月 21, 2025にアクセス、 [https://huggingface.co/docs/transformers/en/main\_classes/trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)  
22. Trainer \- Hugging Face, 10月 21, 2025にアクセス、 [https://huggingface.co/docs/transformers/main\_classes/trainer](https://huggingface.co/docs/transformers/main_classes/trainer)  
23. Training and fine-tuning — transformers 3.1.0 documentation \- Hugging Face, 10月 21, 2025にアクセス、 [https://huggingface.co/transformers/v3.1.0/training.html](https://huggingface.co/transformers/v3.1.0/training.html)  
24. DeBERTa (from the ground up) \+ 2 Approaches \- Kaggle, 10月 21, 2025にアクセス、 [https://www.kaggle.com/code/javigallego/deberta-from-the-ground-up-2-approaches](https://www.kaggle.com/code/javigallego/deberta-from-the-ground-up-2-approaches)  
25. Loss functions. Comprehensive Guide to Loss Functions in Various Machine Learning Domains | by Maxim Sorokin | Medium, 10月 21, 2025にアクセス、 [https://medium.com/@vergotten/loss-functions-comprehensive-guide-to-loss-functions-in-various-machine-learning-domains-1e76f7a9b584](https://medium.com/@vergotten/loss-functions-comprehensive-guide-to-loss-functions-in-various-machine-learning-domains-1e76f7a9b584)  
26. Gradient penalty with respect to the network parameters \- PyTorch Forums, 10月 21, 2025にアクセス、 [https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944](https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944)  
27. reiinakano/invariant-risk-minimization: Implementation of Invariant Risk Minimization https://arxiv.org/abs/1907.02893 \- GitHub, 10月 21, 2025にアクセス、 [https://github.com/reiinakano/invariant-risk-minimization](https://github.com/reiinakano/invariant-risk-minimization)  
28. A Survey on Evaluation of Out-of-Distribution Generalization \- arXiv, 10月 21, 2025にアクセス、 [https://arxiv.org/html/2403.01874v1](https://arxiv.org/html/2403.01874v1)