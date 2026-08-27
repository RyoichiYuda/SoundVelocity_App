# 音速測定アプリ（持ち運び用）

このフォルダーは、音速測定UIの実行に必要なプログラムと既定CSVを
1か所へまとめたものです。リポジトリ内の元ファイルには依存していないため、
`sound_velocity_app` フォルダー全体を別の場所や別PCへコピーして使用できます。

## ファイル構成

```text
sound_velocity_app/
├── main.py                    # 起動入口
├── analysis_pipeline.py       # フィルター・波形照合・音速計算
├── data_sources.py            # CSV読み込み
├── window_functions.py        # 窓関数の生成・保存
├── picoscope_acquisition.py   # PicoScope波形取得
├── requirements.txt           # Python依存パッケージ
└── data/
    ├── hand-0001.csv          # 既定の測定CSV
    └── zero_ref.csv           # 既定の参照CSV
```

## 初回セットアップ

Python 3.12以降を用意し、このフォルダー内で仮想環境と依存パッケージを
セットアップします。

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

LinuxでTkinterがPythonに同梱されていない場合は、ディストリビューションの
パッケージ管理機能で `python3-tk` などもインストールしてください。

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 起動

仮想環境を有効にした状態で実行します。

```bash
python main.py
```

`main.py` を絶対パスで指定すれば、このフォルダー以外をカレント
ディレクトリにして起動しても動作します。画面に入力する相対CSVパスは、
カレントディレクトリではなく、このフォルダーを基準に解決されます。

## PicoScopeを使用する場合

`requirements.txt` に含まれる `pypicosdk` に加えて、使用するOSへ
Pico Technology公式のPicoSDKおよびPicoScope 5000 Series用ドライバーを
インストールしてください。CSVを使った解析テストでは実機接続は不要です。
