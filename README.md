# 研究室のPythonテンプレート

このリポジトリは，研究室の **pythonの機械学習プロジェクト用テンプレート**です．  

採用ツール:

- パッケージ管理: **uv**
- Linter/Formatter: **ruff**
- テスト: **pytest**

---

## 特徴

- `.vscode/`に設定をしているので，`VS Code`でのデバッグの設定，`ruff`による保存時の自動整形の設定がなされています．
- `.github/`には，copilotによるコードレビューに関する設定がなされています．
- `uv`によるパッケージ管理（テンプレートでは，`pytest`と`ruff`だけ入ってます）
- 機械学習プロジェクト向けのディレクトリ構造

---

## 使い方

1. 新しくリポジトリを作るタイミングで，GitHub の **`Start with a template`** からこのテンプレートを選んでリポジトリを作成する。
2. 作成したリポジトリを clone する。
3. `pyproject.toml` の `[project]` セクションの `name`, `description`, `authors` などをプロジェクト用に変更する。
4. `src/lab_template_python` の `lab_template_python` を適切なパッケージ名に変更する。
5. （Python のバージョンを変更する場合）
   - `pyproject.toml` の `[project]` の `requires-python` を変更する  
     （例: Python 3.10 を使いたいなら `">=3.10"` に変更）
   - `pyproject.toml` の `[tool.ruff]` の `target-version` を変更する  
     （例: `py310`）
   - `.python-version` の中身を使用するバージョンに変更する  
     （例: `3.10`）
6. ローカルマシンに`uv` をインストールしていない場合は，先にインストールする。
7. `uv sync --dev` を実行して環境を作成する。

---

## ディレクトリ全体像

```text
repo/
├── configs/                 # 設定ファイル (以下の構造はあくまで一例です)
│   ├── data_generation/     # データ生成用のconfig
│   ├── experiments/         # 実験ごとのconfig
│   │   ├── expXX_name.yaml
│   │   └── expYY_name.yaml
│   └── training/            # 学習用のconfig
│
├── data/                    # ローカルデータ（git管理しない）
│   ├── processed/           # 前処理ずみデータ
│   ├── raw/                 # 生データ
│   └── simulation/          # simulationデータ
│
├── docs/                    # ドキュメントファイル
│   ├── experiments/         # 実験ごとのドキュメントファイル
│
├── models/                  # 学習済みのモデル（git管理しない）
│
├── outputs/                 # 実験結果（git管理しない）
│   ├── canonical/           # 論文/発表で参照する本番実験
│   ├── scratch/             # 試行錯誤・デバッグ（捨てても良い）
│   └── tuning/              # チューニング用
│
├── scripts/                 # 実行スクリプト（以下の構造はあくまで例です）
│   ├── data_generation/     # データの整形・シミュレーションデータの作成用スクリプト
│   ├── evaluation/          # 評価用のスクリプト
│   ├── experiments/         # 実験を回す用のスクリプト
│   │   └── expXX_name/
│   │       ├── README.md
│   │       ├── run.py
│   │       └── sweep.py
│   ├── plots/               # 実験結果の可視化用スクリプト
│   ├── tools/               # ユーティリティ群
│   └── training/            # 学習用のスクリプト
│
├── src/                     # 再利用するライブラリコード (srcレイアウトを採用)
│   └── my_project/          # 適切なプロジェクト名に変更する（以下の構造はあくまで例です）
│       ├── analysis/        # 結果集計ユーティリティ
│       ├── data/            # データ読み込み/前処理
│       ├── models/          # モデル定義
│       ├── simulation/      # シミュレーション用ロジック
│       ├── training/        # 学習ループ，評価指標，評価ロジック
│       ├── utils/           # seed，IOなど
│       └── viz/             # 共通の描画関数/スタイル
│
├── tests/                   # testコード (pytestを使う)
│
├── .gitignore
├── .python-version
├── pyproject.toml           # プロジェクト設定 + ruff/pytest設定
├── README.md
└── uv.lock                  # 依存ロック（コミットする）
```

