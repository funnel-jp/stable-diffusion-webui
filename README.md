# Stable Diffusion web UI
本家（ https://github.com/AUTOMATIC1111/stable-diffusion-webui ）の1.10.1を基に自分用に改変中

## 変更箇所
https://github.com/hako-mikan/sd-webui-lora-block-weight の階層別の重みづけをワンクリックで呼び出せるように以下のファイルを変更した。

ui_extra_networks_lora.py（stable-diffusion-webui/extensions-builtin/Lora/）

ui_edit_user_metadata.py（stable-diffusion-webui/extensions-builtin/Lora/）

extraNetworks.js（stable-diffusion-webui/javascript/）
