# Stable Diffusion web UI
本家（ https://github.com/AUTOMATIC1111/stable-diffusion-webui ）の1.10.1を基に自分用に改変中

## 変更箇所
https://github.com/hako-mikan/sd-webui-lora-block-weight の階層別の重みづけをワンクリックで呼び出せるように以下のファイルを変更した。

ui_extra_networks_lora.py（stable-diffusion-webui/extensions-builtin/Lora/）

ui_edit_user_metadata.py（stable-diffusion-webui/extensions-builtin/Lora/）

extraNetworks.js（stable-diffusion-webui/javascript/）

## 導入方法
既存のstable-diffusion-webuiのフォルダで以下のコマンドを順番に実行してください。

git remote add fork_lora_block_weight https://github.com/funnel-jp/stable-diffusion-webui.git

git fetch fork_lora_block_weight

git checkout fork_lora_block_weight/master -- extensions-builtin/Lora/ui_extra_networks_lora.py extensions-builtin/Lora/ui_edit_user_metadata.py javascript/extraNetworks.js

元に戻すときは

git fetch origin

git checkout origin/master -- extensions-builtin/Lora/ui_extra_networks_lora.py extensions-builtin/Lora/ui_edit_user_metadata.py javascript/extraNetworks.js

