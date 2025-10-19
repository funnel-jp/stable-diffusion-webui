
# (ここにリポジトリ名を入力) - Stable Diffusion WebUI カスタム版

このリポジトリは、[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (本家) に対し、いくつかの利便性向上機能を追加した変更版です。

本家webuiを利用している環境に `git remote` を使って導入し、ブランチを切り替えるだけで本家環境とこのカスタム環境を簡単に併用できます。

---

## 主な機能

### 1. LoRA階層別重みづけ（Block Weight）の簡易呼び出し

[hako-mikan/sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) で実装されているLoRAの階層別重みづけ指定（例: `ALL,INS,MIDD,OUTDや0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0`）を、Extra NetworksのUI上からワンクリックでプロンプトに挿入できるようにしました。

これにより、複雑な重みづけを手入力する手間が省けます。

**関連ファイル:**
* `extensions-builtin/Lora/ui_extra_networks_lora.py`
* `extensions-builtin/Lora/ui_edit_user_metadata.py`
* `javascript/extraNetworks.js`

### 2. XYZプロットのグリッド分割出力

標準の `scripts/xyz_grid.py` （X/Y/Z plot）において、**バッチサイズ (Batch size) を2以上に設定**して画像生成を行った際の動作を変更しました。

* **変更前:** バッチサイズ分の画像がすべて連結された、1枚の巨大なグリッド画像が生成される。
* **変更後:** バッチごと（バッチサイズ分）にグリッド画像を分割して生成します（「バッチ数 (Batch count) を増やした」時と同様の挙動）。

これにより、FP8生成などと組み合わせてプロンプト比較のための生成時間を短縮できる可能性があります。
**関連ファイル:**
* `scripts/xyz_grid.py`

---

## 導入方法

すでに本家 `AUTOMATIC1111/stable-diffusion-webui` を `git clone` して使用している環境への導入を想定しています。

**（重要）**
以下の手順を実行する前に、ご自身の `stable-diffusion-webui` フォルダで、変更したファイル（`ui_*.py`, `extraNetworks.js`, `xyz_grid.py`）をコミットしていないか、`git status` コマンドで確認してください。もし変更がある場合は、`git stash` で一時退避するか、コミットしてください。

---

### ステップ1: このリポジトリをリモートに追加

お使いの `stable-diffusion-webui` ディレクトリに移動し、`git remote` コマンドを実行して、このリポジトリを `funnel` という名前（任意）でリモートリポジトリとして登録します。

```bash
# stable-diffusion-webui ディレクトリに移動
cd path/to/stable-diffusion-webui

# リモートリポジトリ 'funnel' を追加
git remote add funnel https://github.com/funnel-jp/stable-diffusion-webui.git

# 登録されたか確認
git remote -v

````
*(実行例)*
```
origin    [https://github.com/AUTOMATIC1111/stable-diffusion-webui.git](https://github.com/AUTOMATIC1111/stable-diffusion-webui.git) (fetch)
origin    [https://github.com/AUTOMATIC1111/stable-diffusion-webui.git](https://github.com/AUTOMATIC1111/stable-diffusion-webui.git) (push)
funnel    [https://github.com/funnel-jp/stable-diffusion-webui.git](https://github.com/funnel-jp/stable-diffusion-webui.git) (fetch)
funnel    [https://github.com/funnel-jp/stable-diffusion-webui.git](https://github.com/funnel-jp/stable-diffusion-webui.git) (push)
````

### ステップ2: 変更内容の取得

追加したリモート（`funnel`）から変更履歴を取得（fetch）します。

```bash
git fetch funnel
```

### ステップ3: カスタムブランチへの切り替え

取得したブランチに切り替えます。（このリポジトリのブランチ名を `main` とします）

```bash
# 'funnel' リモートの 'main' ブランチに切り替える
git switch funnel

# もしローカルに新しいブランチ 'funnel-features' を作って切り替える場合
# git switch -c funnel-features funnel/main
```

以上で導入は完了です。`webui.sh` または `webui.bat` を起動すると、上記の機能が適用された状態で使用できます。
