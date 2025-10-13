# Stable Diffusion web UI
本家（ https://github.com/AUTOMATIC1111/stable-diffusion-webui ）の1.10.1を基に自分用に改変中

## 変更箇所
https://github.com/hako-mikan/sd-webui-lora-block-weight の階層別の重みづけをワンクリックで呼び出せるように以下のファイルを変更した。

ui_extra_networks_lora.py（stable-diffusion-webui/extensions-builtin/Lora/）

ui_edit_user_metadata.py（stable-diffusion-webui/extensions-builtin/Lora/）

extraNetworks.js（stable-diffusion-webui/javascript/）

xyzプロットで、バッチサイズを上げた際に巨大なグリッドではなく、バッチ数分のグリッドを分割して作成するようにした。

xyz_grid.py（stable-diffusion-webui/scripts/）

## 導入方法
既存のstable-diffusion-webuiのフォルダで以下のコマンドを順番に実行してください。

git remote add fork_lora_block_weight https://github.com/funnel-jp/stable-diffusion-webui.git

git fetch fork_funnel

git checkout fork_funnel/master -- extensions-builtin/Lora/ui_extra_networks_lora.py extensions-builtin/Lora/ui_edit_user_metadata.py javascript/extraNetworks.js scripts/xyz_grid.py

元に戻すときは

git fetch origin

git checkout origin/master -- extensions-builtin/Lora/ui_extra_networks_lora.py extensions-builtin/Lora/ui_edit_user_metadata.py javascript/extraNetworks.js

## 使い方
LoRAタブでモデルカードの設定を開くとadditional weight欄が追加されています。この欄にALLや1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0 のように入力して保存するとjsonファイルに"additional weight"が記録されます。この状態でLoRAを呼び出すとプロンプトに自動で:lbw="additional weight"が追加されます。

xyzプロットのUIにSplit grids by batch sizeが追加されているので、これを有効にするとバッチサイズを2以上にした際の動作が変更され、バッチの数だけ個別のグリッド画像（Batch #1, Batch #2, ...）を生成します。また、Legend for Zをオフにすると、邪魔なBatch #Xのラベルを非表示にして画像をより大きく表示できます。


