# This script is a modified version of xyz_grid.py from the AUTOMATIC1111's stable-diffusion-webui.
# Original source: https://github.com/AUTOMATIC1111/stable-diffusion-webui
#
# --- このスクリプトへの改造点 ---
# 1. Split grids by batch size:
#    バッチサイズ（Batch size > 1）に応じて、グリッド画像をバッチの数だけ個別に生成する機能を追加。
# 2. Per-axis legend visibility:
#    凡例（X/Y/Z軸のラベル）を、軸ごとに個別に表示/非表示に設定できる機能を追加。
# 3. "Include Sub Grids" compatibility:
#    上記のバッチ分割機能と "Include Sub Grids" 機能を併用できるようにし、
#    バッチごと、かつZ軸の値ごとにXYサブグリッドをすべて出力できるようにした。
# 4. Dynamic Prompts compatibility:
#    Dynamic Prompts（ワイルドカード）使用時に、メタデータにワイルドカード展開後のプロンプトが
#    正しく記録されるように、`all_prompts`の収集ロジックを修正。
# 5. Various bug fixes:
#    片方の軸の凡例だけをオフにした場合のエラーや、メタデータの不整合による
#    画像保存時のエラーなどを修正。
# ---

from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
import os.path
from io import StringIO
from PIL import Image
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images, sd_samplers, processing, sd_models, sd_vae, sd_schedulers, errors
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import re

from modules.ui_components import ToolButton, InputAccordion

# UI用のアイコン
fill_values_symbol = "\U0001f4d2"  # 📒

# 軸の情報を保持するための名前付きタプル
AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])


# --- 軸の適用関数 ---
# 各軸のタイプに応じて、画像生成パラメータ(p)に値を適用するためのヘルパー関数群

def apply_field(field):
    """
    pオブジェクトの指定されたフィールド(属性)に値を設定する関数を返す。
    例: apply_field("seed") は p.seed = x を実行する関数を返す。
    """
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    """
    プロンプト内の特定の文字列(xs[0])を、現在の軸の値(x)で置き換える。
    ネガティブプロンプトも同様に置き換える。
    """
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    """
    「Prompt order」軸用。プロンプト内のトークンの順序を並べ替える。
    """
    token_order = []

    # Initially grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def confirm_samplers(p, xs):
    """ サンプラー名が実在するか確認する """
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise RuntimeError(f"Unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    """ チェックポイント(モデル)を適用する """
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    p.override_settings['sd_model_checkpoint'] = info.name


def confirm_checkpoints(p, xs):
    """ チェックポイント名が実在するか確認する """
    for x in xs:
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def confirm_checkpoints_or_none(p, xs):
    """ チェックポイント名が実在するか、あるいは "None" かを確認する (Refiner用) """
    for x in xs:
        if x in (None, "", "None", "none"):
            continue

        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def confirm_range(min_val, max_val, axis_label):
    """
    値が指定された範囲内にあるか確認する関数を生成する。
    """

    def confirm_range_fun(p, xs):
        for x in xs:
            if not (max_val >= x >= min_val):
                raise ValueError(f'{axis_label} value "{x}" out of range [{min_val}, {max_val}]')

    return confirm_range_fun


def apply_size(p, x: str, xs) -> None:
    """ "Width x Height" 形式の文字列を解釈して、p.width と p.height に適用する """
    try:
        width, _, height = x.partition('x')
        width = int(width.strip())
        height = int(height.strip())
        p.width = width
        p.height = height
    except ValueError:
        print(f"Invalid size in XYZ plot: {x}")


def find_vae(name: str):
    """ VAE名から、内部的なVAEオブジェクトを検索する """
    if (name := name.strip().lower()) in ('auto', 'automatic'):
        return 'Automatic'
    elif name == 'none':
        return 'None'
    return next((k for k in modules.sd_vae.vae_dict if k.lower() == name), print(f'No VAE found for {name}; using Automatic') or 'Automatic')


def apply_vae(p, x, xs):
    """ VAEを適用する """
    p.override_settings['sd_vae'] = find_vae(x)


def apply_styles(p: StableDiffusionProcessingTxt2Img, x: str, _):
    """ スタイルを適用する (カンマ区切りで複数指定可能) """
    p.styles.extend(x.split(','))


def apply_uni_pc_order(p, x, xs):
    """ UniPC Samplerの次数を適用する """
    p.override_settings['uni_pc_order'] = min(x, p.steps - 1)


def apply_face_restore(p, opt, x):
    """ 顔修復(Face restore)を適用する """
    opt = opt.lower()
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    else:
        is_active = opt in ('true', 'yes', 'y', '1')

    p.restore_faces = is_active


def apply_override(field, boolean: bool = False):
    """
    p.override_settings を通じて、Web UIの内部設定値を上書きする。
    """
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        p.override_settings[field] = x

    return fun


def boolean_choice(reverse: bool = False):
    """ Boolean型(True/False)の選択肢を返す """
    def choice():
        return ["False", "True"] if reverse else ["True", "False"]

    return choice


# --- 値のフォーマット関数 ---
# グリッドの凡例(ラベル)に表示するテキストを整形するためのヘルパー関数群

def format_value_add_label(p, opt, x):
    """ "Label: Value" の形式で返す (例: "Seed: 1234") """
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    """ 値をそのまま返す (例: "1234") """
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    """ リストの値をカンマ区切りで結合して返す (Prompt order用) """
    return ", ".join(x)


def do_nothing(p, x, xs):
    """ 「Nothing」軸用。何もしない。 """
    pass


def format_nothing(p, opt, x):
    """ 「Nothing」軸用。凡例に何も表示しない。 """
    return ""


def format_remove_path(p, opt, x):
    """ ファイルパスからファイル名のみを抽出して返す (Checkpoint, VAE用) """
    return os.path.basename(x)


def str_permutations(x):
    """ 「Prompt order」軸用のダミー型。順列を扱うことを示す。 """
    return x


# --- CSV/リスト変換 ヘルパー関数 ---

def list_to_csv_string(data_list):
    """ 文字列のリストをCSV形式の単一の文字列に変換する """
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()


def csv_string_to_list_strip(data_str):
    """ CSV形式の文字列を、前後の空白を除去した文字列のリストに変換する """
    return list(map(str.strip, chain.from_iterable(csv.reader(StringIO(data_str), skipinitialspace=True))))


# --- 軸の定義 ---

class AxisOption:
    """
    X/Y/Z軸の各選択肢を定義するクラス。
    label: UIに表示される名前
    type: 値の型 (int, float, strなど)
    apply: 値を適用するための関数
    format_value: 凡例に表示するテキストを整形する関数
    confirm: 値が有効か確認する関数 (オプション)
    cost: この軸を変更するコスト（コストが高い軸が内側のループになるように最適化される）
    choices: 選択肢をドロップダウンで提供する場合の、選択肢を返す関数 (オプション)
    prepare: テキストボックスの入力を処理する関数 (オプション)
    """
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value
        self.confirm = confirm
        self.cost = cost
        self.prepare = prepare
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    """ Img2Imgタブでのみ表示される軸 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True


class AxisOptionTxt2Img(AxisOption):
    """ Txt2Imgタブでのみ表示される軸 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


# UIに表示されるすべての軸のリスト
axis_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    AxisOption("Prompt order", str_permutations, apply_order, format_value=format_value_join_list),
    AxisOptionTxt2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers if x.name not in opts.hide_samplers]),
    AxisOptionTxt2Img("Hires sampler", str, apply_field("hr_sampler_name"), confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    AxisOptionImg2Img("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in opts.hide_samplers]),
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value=format_remove_path, confirm=confirm_checkpoints, cost=1.0, choices=lambda: sorted(sd_models.checkpoints_list, key=str.casefold)),
    AxisOption("Negative Guidance minimum sigma", float, apply_field("s_min_uncond")),
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    AxisOption("Schedule type", str, apply_field("scheduler"), choices=lambda: [x.label for x in sd_schedulers.schedulers]),
    AxisOption("Schedule min sigma", float, apply_override("sigma_min")),
    AxisOption("Schedule max sigma", float, apply_override("sigma_max")),
    AxisOption("Schedule rho", float, apply_override("rho")),
    AxisOption("Skip Early CFG", float, apply_override('skip_early_cond')),
    AxisOption("Beta schedule alpha", float, apply_override("beta_dist_alpha")),
    AxisOption("Beta schedule beta", float, apply_override("beta_dist_beta")),
    AxisOption("Eta", float, apply_field("eta")),
    AxisOption("Clip skip", int, apply_override('CLIP_stop_at_last_layers')),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOption("Initial noise multiplier", float, apply_field("initial_noise_multiplier")),
    AxisOption("Extra noise", float, apply_override("img2img_extra_noise")),
    AxisOptionTxt2Img("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOptionImg2Img("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['Automatic', 'None'] + list(sd_vae.vae_dict)),
    AxisOption("Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)),
    AxisOption("UniPC Order", int, apply_uni_pc_order, cost=0.5),
    AxisOption("Face restore", str, apply_face_restore, format_value=format_value),
    AxisOption("Token merging ratio", float, apply_override('token_merging_ratio')),
    AxisOption("Token merging ratio high-res", float, apply_override('token_merging_ratio_hr')),
    AxisOption("Always discard next-to-last sigma", str, apply_override('always_discard_next_to_last_sigma', boolean=True), choices=boolean_choice(reverse=True)),
    AxisOption("SGM noise multiplier", str, apply_override('sgm_noise_multiplier', boolean=True), choices=boolean_choice(reverse=True)),
    AxisOption("Refiner checkpoint", str, apply_field('refiner_checkpoint'), format_value=format_remove_path, confirm=confirm_checkpoints_or_none, cost=1.0, choices=lambda: ['None'] + sorted(sd_models.checkpoints_list, key=str.casefold)),
    AxisOption("Refiner switch at", float, apply_field('refiner_switch_at')),
    AxisOption("RNG source", str, apply_override("randn_source"), choices=lambda: ["GPU", "CPU", "NV"]),
    AxisOption("FP8 mode", str, apply_override("fp8_storage"), cost=0.9, choices=lambda: ["Disable", "Enable for SDXL", "Enable"]),
    AxisOption("Size", str, apply_size),
]


def draw_xyz_grid(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, draw_legend, include_lone_images, include_sub_grids, first_axes_processed, second_axes_processed, margin_size, draw_grid):
    """
    オリジナルのグリッド描画関数。
    "Split grids by batch size" がオフの時に呼び出される。
    バッチサイズ > 1 の場合、各セルの1枚目の画像のみを使ってグリッドを構築する。
    """
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    title_texts = [[images.GridAnnotation(z)] for z in z_labels]

    list_size = (len(xs) * len(ys) * len(zs))

    processed_result = None

    state.job_count = list_size * p.n_iter

    def process_cell(x, y, z, ix, iy, iz):
        """
        X/Y/Zの各組み合わせ（セル）の画像を1つ生成する内部関数。
        """
        nonlocal processed_result

        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        state.job = f"{index(ix, iy, iz) + 1} out of {list_size}"

        # cell()は実際にはrun()メソッド内で定義されたcell関数を指す
        processed: Processed = cell(x, y, z, ix, iy, iz)

        if processed_result is None:
            # 最初の結果をテンプレートとして、最終的な結果を格納するProcessedオブジェクトを作成
            processed_result = copy(processed)
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.index_of_first_image = 1

        idx = index(ix, iy, iz)
        if processed.images:
            # バッチサイズが > 1 であっても、[0]番目（1枚目）の画像のみをグリッドに使用する
            processed_result.images[idx] = processed.images[0]
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
        else:
            # 画像生成に失敗した場合、空の画像で埋める
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)

    # 軸のコストに基づいて、最も効率的な順序でループを回す
    if first_axes_processed == 'x':
        for ix, x in enumerate(xs):
            if second_axes_processed == 'y':
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == 'y':
        for iy, y in enumerate(ys):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == 'z':
        for iz, z in enumerate(zs):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iy, y in enumerate(ys):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)

    if not processed_result:
        print("Unexpected error: Processing could not begin, you may need to refresh the tab or restart the service.")
        return Processed(p, [])
    elif not any(processed_result.images):
        print("Unexpected error: draw_xyz_grid failed to return even a single processed image")
        return Processed(p, [])

    # グリッド描画が有効な場合
    if draw_grid:
        z_count = len(zs)

        # Z軸の各値に対応するXYサブグリッドを作成
        for i in range(z_count):
            start_index = (i * len(xs) * len(ys)) + i
            end_index = start_index + len(xs) * len(ys)
            grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(ys))
            if draw_legend:
                grid_max_w, grid_max_h = map(max, zip(*(img.size for img in processed_result.images[start_index:end_index])))
                grid = images.draw_grid_annotations(grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
            # サブグリッドを結果リストの先頭に追加していく
            processed_result.images.insert(i, grid)
            processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
            processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
            processed_result.infotexts.insert(i, processed_result.infotexts[start_index])

        # Z軸のグリッド（サブグリッドをまとめたグリッド）を作成
        z_grid = images.image_grid(processed_result.images[:z_count], rows=1)
        z_sub_grid_max_w, z_sub_grid_max_h = map(max, zip(*(img.size for img in processed_result.images[:z_count])))
        if draw_legend:
            z_grid = images.draw_grid_annotations(z_grid, z_sub_grid_max_w, z_sub_grid_max_h, title_texts, [[images.GridAnnotation()]])
        # 最終的なZグリッドをリストの0番目に追加
        processed_result.images.insert(0, z_grid)
        processed_result.infotexts.insert(0, processed_result.infotexts[0])

    return processed_result

# --- ▼▼▼ ここからが改造された機能 ▼▼▼ ---

def draw_xyz_grid_split_by_batch(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, margin_size, draw_legend_x, draw_legend_y, draw_legend_z, include_sub_grids):
    """
    【新機能】バッチサイズ分割モード用のグリッド描画関数。
    - バッチサイズNに応じて、N個の独立したグリッド画像を生成する。
    - "Include Sub Grids" が有効な場合、バッチごと・Z軸スライスごとのXYサブグリッドもすべて出力する。
    - 軸ごとの凡例表示/非表示に対応する。
    - Dynamic Prompts（ワイルドカード）展開後のプロンプトをメタデータに正しく記録する。
    """
    
    batch_size = p.batch_size
    # 1つのグリッドあたり（1バッチあたり）の総画像数
    images_per_grid = len(xs) * len(ys) * len(zs)
    
    # UIに進捗バーの総ステップ数を設定
    state.job_count = images_per_grid * p.n_iter

    # --- 1. 全ての個別画像を収集する ---
    
    processed_template = None  # 最初のProcessedオブジェクトをテンプレートとして保存
    
    # バッチの数だけ、結果を格納するリスト（Processedオブジェクト）を用意
    # 例: batch_size=2 なら [Processed_for_batch_1, Processed_for_batch_2]
    batch_processed_results = []

    # X, Y, Zの3重ループで、すべてのセルを処理
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                # UIに進捗を表示
                state.job = f"{ix + iy * len(xs) + iz * len(xs) * len(ys) + 1} out of {images_per_grid}"
                
                # cell()を呼び出し、バッチサイズ分の画像を一括生成
                # (p.batch_size = N の場合、 processed_cell_result.images には N+1 枚の画像が入る)
                # [0] = バッチN枚をまとめたグリッド画像
                # [1] = バッチ1枚目の個別画像
                # [2] = バッチ2枚目の個別画像
                # ...
                processed_cell_result = cell(x, y, z, ix, iy, iz)

                # 最初のセル処理時に、結果格納用のコンテナを初期化
                if processed_template is None:
                    # 最初のセルが失敗（画像が2枚未満）した場合はエラー
                    if not processed_cell_result.images or len(processed_cell_result.images) < 2:
                        raise RuntimeError(f"The first cell failed to produce a batch of images. Expected at least 2 images (grid + 1), but got {len(processed_cell_result.images)}.")
                    
                    # 最初の結果をテンプレートとしてコピー
                    processed_template = copy(processed_cell_result)
                    
                    # 個別画像(images[1])のサイズとモードを基準に、プレースホルダー画像を作成
                    single_image_mode = processed_template.images[1].mode
                    single_image_size = processed_template.images[1].size
                    
                    # バッチサイズ分ループし、空の結果コンテナを作成
                    for i in range(batch_size):
                        res = copy(processed_template)
                        # images_per_grid の数だけ、空の画像で埋めたリストを作成
                        res.images = [Image.new(single_image_mode, single_image_size)] * images_per_grid
                        res.all_prompts = [""] * images_per_grid
                        res.all_negative_prompts = [""] * images_per_grid
                        res.all_seeds = [-1] * images_per_grid
                        res.all_subseeds = [-1] * images_per_grid
                        res.infotexts = [""] * images_per_grid
                        batch_processed_results.append(res)

                # 画像生成に成功した場合
                if processed_cell_result.images:
                    # このセルがグリッド全体（1バッチ分）のどこに位置するかを計算
                    grid_index = ix + iy * len(xs) + iz * len(xs) * len(ys)
                    
                    # [0]番目の自動生成グリッドを無視し、[1]番目以降の個別画像のみを取得
                    individual_images = processed_cell_result.images[1:]
                    
                    # バッチサイズ分ループし、各バッチコンテナに画像とメタデータを振り分ける
                    for i in range(batch_size):
                        if i < len(individual_images):
                            # バッチi番目のコンテナの、正しい位置(grid_index)に画像iを格納
                            batch_processed_results[i].images[grid_index] = individual_images[i]
                            
                            # --- Dynamic Prompts対応 ---
                            # processed_cell_result.all_prompts には、ワイルドカード展開後のプロンプトが
                            # バッチサイズ分格納されている。これを正しく収集する。
                            if i < len(processed_cell_result.all_prompts):
                                batch_processed_results[i].all_prompts[grid_index] = processed_cell_result.all_prompts[i]
                            else:
                                batch_processed_results[i].all_prompts[grid_index] = processed_cell_result.prompt # 予備

                            # 他メタデータも同様に収集
                            batch_processed_results[i].all_seeds[grid_index] = processed_cell_result.all_seeds[i]
                            batch_processed_results[i].infotexts[grid_index] = processed_cell_result.infotexts[i]
                            if i < len(processed_cell_result.all_negative_prompts):
                                batch_processed_results[i].all_negative_prompts[grid_index] = processed_cell_result.all_negative_prompts[i]
                            if i < len(processed_cell_result.all_subseeds):
                                batch_processed_results[i].all_subseeds[grid_index] = processed_cell_result.all_subseeds[i]

    # --- 2. 収集した個別画像から、グリッドとサブグリッドを組み立てる ---
    
    final_z_grids = []                # Z軸でまとめた最終グリッド (バッチの数だけ)
    final_z_grids_infotexts = []
    final_xy_sub_grids = []           # XYサブグリッド (include_sub_gridsがONの場合)
    final_xy_sub_grids_infotexts = []

    # バッチごとの結果コンテナ (batch_processed_results) をループ
    for i, res in enumerate(batch_processed_results):
        valid_images = [img for img in res.images if img is not None]
        if not valid_images: continue # 画像が1枚もない場合はスキップ

        # Z軸が設定されている場合 (Z軸の値が複数ある場合)
        if len(zs) > 1:
            z_slice_grids = [] # バッチiの、Z軸スライスごとのXYグリッドを一時的に格納
            
            # Z軸の各値についてループ
            for iz in range(len(zs)):
                start = iz * len(xs) * len(ys)
                end = start + len(xs) * len(ys)
                
                # Z軸の現在の値(iz)に対応するXYサブグリッドを生成
                sub_grid = images.image_grid(res.images[start:end], rows=len(ys))
                
                # --- 凡例（ラベル）の描画処理 ---
                if draw_legend_x or draw_legend_y:
                    # 【エラー修正】片方の軸の凡例がオフでも、もう片方がオンなら描画関数を呼ぶ。
                    # その際、オフの軸には「空のラベル」リストを作成して渡す。
                    hor_texts = [[images.GridAnnotation(x)] for x in x_labels] if draw_legend_x else [[images.GridAnnotation()] for _ in x_labels]
                    ver_texts = [[images.GridAnnotation(y)] for y in y_labels] if draw_legend_y else [[images.GridAnnotation()] for _ in y_labels]
                    
                    valid_sub_images = [img for img in res.images[start:end] if img is not None]
                    if valid_sub_images:
                        grid_max_w, grid_max_h = map(max, zip(*(img.size for img in valid_sub_images)))
                        # XYサブグリッドに凡例を描画
                        sub_grid = images.draw_grid_annotations(sub_grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
                
                z_slice_grids.append(sub_grid) # Zグリッド生成用に一時保存

                # --- "Include Sub Grids" 機能 ---
                if include_sub_grids:
                    # 生成したXYサブグリッドを最終結果リストに追加
                    final_xy_sub_grids.append(sub_grid)
                    
                    # サブグリッド用のメタデータを作成
                    info_p = copy(p)
                    info_p.extra_generation_params = copy(p.extra_generation_params)
                    info_p.extra_generation_params["Batch grid index"] = i + 1
                    info_p.extra_generation_params["Z-Value"] = z_labels[iz] # どのZ軸の値か明記
                    
                    # メタデータには、そのグリッドの先頭画像(start)の情報を代表として使用
                    sub_grid_infotext = processing.create_infotext(info_p, [res.all_prompts[start]], [res.all_seeds[start]], [res.all_subseeds[start]], all_negative_prompts=[res.all_negative_prompts[start]])
                    final_xy_sub_grids_infotexts.append(sub_grid_infotext)

            # Z軸をまとめた最終グリッド (XYサブグリッドを横に並べる) を生成
            z_grid = images.image_grid(z_slice_grids, rows=1)
            
            # Z軸の凡例を描画（"Batch #X" ラベルもここで描画）
            z_texts = [[images.GridAnnotation(z)] for z in z_labels] if draw_legend_z else []
            batch_texts = [[images.GridAnnotation(f"Batch #{i+1}")]] if draw_legend_z else []
            if draw_legend_z: # Z軸凡例がオンの場合のみ描画
                valid_z_grids = [img for img in z_slice_grids if img is not None]
                if valid_z_grids:
                    grid_max_w, grid_max_h = map(max, zip(*(img.size for img in valid_z_grids)))
                    z_grid = images.draw_grid_annotations(z_grid, grid_max_w, grid_max_h, z_texts, batch_texts)
            
            final_z_grids.append(z_grid)
            # Zグリッドの代表メタデータを作成
            z_grid_infotext = processing.create_infotext(p, [res.all_prompts[0]], [res.all_seeds[0]], [res.all_subseeds[0]], all_negative_prompts=[res.all_negative_prompts[0]])
            final_z_grids_infotexts.append(z_grid_infotext)

        else: # Z軸が設定されていない場合
            # XYグリッドがそのまま最終グリッドになる
            grid = images.image_grid(res.images, rows=len(ys))
            
            # X, Y軸の凡例を描画（片方オフでもOKなように修正済み）
            if draw_legend_x or draw_legend_y:
                hor_texts = [[images.GridAnnotation(x)] for x in x_labels] if draw_legend_x else [[images.GridAnnotation()] for _ in x_labels]
                ver_texts = [[images.GridAnnotation(y)] for y in y_labels] if draw_legend_y else [[images.GridAnnotation()] for _ in y_labels]

                grid_max_w, grid_max_h = map(max, zip(*(img.size for img in valid_images)))
                grid = images.draw_grid_annotations(grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)

            final_z_grids.append(grid)
            # メタデータを作成
            z_grid_infotext = processing.create_infotext(p, [res.all_prompts[0]], [res.all_seeds[0]], [res.all_subseeds[0]], all_negative_prompts=[res.all_negative_prompts[0]])
            final_z_grids_infotexts.append(z_grid_infotext)


    # --- 3. 最終的な結果を統合して返す ---
    
    if not processed_template:
        return Processed(p, []) # 1枚も画像が生成されなかった場合

    final_processed = processed_template
    
    # 最終的な画像リスト = [Zグリッド(バッチ分)] + [XYサブグリッド(バッチ*Z軸分)]
    final_processed.images = final_z_grids + final_xy_sub_grids
    # メタデータも同じ順序で結合
    final_processed.infotexts = final_z_grids_infotexts + final_xy_sub_grids_infotexts
    
    # all_promptsとall_seedsは、画像保存時やUI表示時に「代表値」として使われる
    # ここには、各バッチの最初の画像(index 0)の「解決済みプロンプト」を格納する
    final_processed.all_prompts = [res.all_prompts[0] for res in batch_processed_results]
    final_processed.all_seeds = [res.all_seeds[0] for res in batch_processed_results]
    
    # 【重要】UIがすべての画像を「個別の画像」としてではなく、
    # 「すべてが主要なグリッド画像」として扱うように設定
    final_processed.index_of_first_image = 0
    return final_processed

# --- ▲▲▲ 改造された機能はここまで ▲▲▲ ---


class SharedSettingsStackHelper(object):
    """
    設定を一時的に変更し、処理後に元に戻すためのヘルパークラス。
    (VAEやモデルの切り替えなどで使用)
    """
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        modules.sd_models.reload_model_weights()
        modules.sd_vae.reload_vae_weights()


# テキストボックスの "1-5" や "1-10(2)" のような範囲指定を解釈するための正規表現
re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*])?\s*")


class Script(scripts.Script):
    """
    Web UIのスクリプトとして登録されるメインクラス。
    """
    
    def title(self):
        """ スクリプトのタイトルを返す """
        return "X/Y/Z plot"

    def ui(self, is_img2img):
        """
        スクリプトのUI（ドロップダウンやチェックボックスなど）を構築する。
        """
        
        # 現在のタブ(txt2img / img2img)に応じて、表示する軸の選択肢をフィルタリング
        self.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]

        with gr.Row():
            with gr.Column(scale=19):
                with gr.Row():
                    # X軸のUI
                    x_type = gr.Dropdown(label="X type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[1].label, type="index", elem_id=self.elem_id("x_type"))
                    x_values = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"))
                    x_values_dropdown = gr.Dropdown(label="X values", visible=False, multiselect=True, interactive=True)
                    fill_x_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_x_tool_button", visible=False)

                with gr.Row():
                    # Y軸のUI
                    y_type = gr.Dropdown(label="Y type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                    y_values = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"))
                    y_values_dropdown = gr.Dropdown(label="Y values", visible=False, multiselect=True, interactive=True)
                    fill_y_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_y_tool_button", visible=False)

                with gr.Row():
                    # Z軸のUI
                    z_type = gr.Dropdown(label="Z type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("z_type"))
                    z_values = gr.Textbox(label="Z values", lines=1, elem_id=self.elem_id("z_values"))
                    z_values_dropdown = gr.Dropdown(label="Z values", visible=False, multiselect=True, interactive=True)
                    fill_z_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_z_tool_button", visible=False)

        with gr.Row(variant="compact", elem_id="axis_options"):
            with gr.Column():
                no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))
                with gr.Row():
                    vary_seeds_x = gr.Checkbox(label='Vary seeds for X', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_x"), tooltip="Use different seeds for images along X axis.")
                    vary_seeds_y = gr.Checkbox(label='Vary seeds for Y', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_y"), tooltip="Use different seeds for images along Y axis.")
                    vary_seeds_z = gr.Checkbox(label='Vary seeds for Z', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_z"), tooltip="Use different seeds for images along Z axis.")
            with gr.Column():
                include_lone_images = gr.Checkbox(label='Include Sub Images', value=False, elem_id=self.elem_id("include_lone_images"))
                csv_mode = gr.Checkbox(label='Use text inputs instead of dropdowns', value=False, elem_id=self.elem_id("csv_mode"))
                
                # --- ▼▼▼ 新機能UI ▼▼▼ ---
                split_grids_by_batch = gr.Checkbox(label='Split grids by batch size', value=False, elem_id=self.elem_id("split_grids_by_batch"), tooltip="Instead of one grid, create N grids for batch size N.")
                # --- ▲▲▲ 新機能UI ▲▲▲ ---

        with InputAccordion(True, label='Draw grid', elem_id=self.elem_id('draw_grid')) as draw_grid:
            with gr.Row():
                include_sub_grids = gr.Checkbox(label='Include Sub Grids', value=False, elem_id=self.elem_id("include_sub_grids"))
                margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))

            # --- ▼▼▼ 新機能UI ▼▼▼ ---
            # 元の "Draw legend" チェックボックスを、X, Y, Z軸ごとに分割
            with gr.Row(elem_id="legend_options"):
                draw_legend_x = gr.Checkbox(label='Legend for X', value=True, elem_id=self.elem_id("draw_legend_x"))
                draw_legend_y = gr.Checkbox(label='Legend for Y', value=True, elem_id=self.elem_id("draw_legend_y"))
                draw_legend_z = gr.Checkbox(label='Legend for Z', value=True, elem_id=self.elem_id("draw_legend_z"))
            # --- ▲▲▲ 新機能UI ▲▲▲ ---

        with gr.Row(variant="compact", elem_id="swap_axes"):
            swap_xy_axes_button = gr.Button(value="Swap X/Y axes", elem_id="xy_grid_swap_axes_button")
            swap_yz_axes_button = gr.Button(value="Swap Y/Z axes", elem_id="yz_grid_swap_axes_button")
            swap_xz_axes_button = gr.Button(value="Swap X/Z axes", elem_id="xz_grid_swap_axes_button")

        # --- UIのコールバック関数 (UIの操作に応じて動的に表示を変更) ---
        
        def swap_axes(axis1_type, axis1_values, axis1_values_dropdown, axis2_type, axis2_values, axis2_values_dropdown):
            """ 軸の入れ替えボタンが押されたときの処理 """
            return self.current_axis_options[axis2_type].label, axis2_values, axis2_values_dropdown, self.current_axis_options[axis1_type].label, axis1_values, axis1_values_dropdown

        xy_swap_args = [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown]
        swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
        yz_swap_args = [y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
        xz_swap_args = [x_type, x_values, x_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

        def fill(axis_type, csv_mode):
            """ 📒 (Fill) ボタンが押されたときの処理。ドロップダウンの選択肢を値テキストボックスに書き込む """
            axis = self.current_axis_options[axis_type]
            if axis.choices:
                if csv_mode:
                    return list_to_csv_string(axis.choices()), gr.update()
                else:
                    return gr.update(), axis.choices()
            else:
                return gr.update(), gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values, x_values_dropdown])
        fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values, y_values_dropdown])
        fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values, z_values_dropdown])

        def select_axis(axis_type, axis_values, axis_values_dropdown, csv_mode):
            """ 軸のタイプ（ドロップダウン）が変更されたときの処理。値の入力欄をテキストボックス/ドロップダウンに切り替える """
            axis_type = axis_type or 0

            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None

            if has_choices:
                choices = choices()
                if csv_mode:
                    if axis_values_dropdown:
                        axis_values = list_to_csv_string(list(filter(lambda x: x in choices, axis_values_dropdown)))
                        axis_values_dropdown = []
                else:
                    if axis_values:
                        axis_values_dropdown = list(filter(lambda x: x in choices, csv_string_to_list_strip(axis_values)))
                        axis_values = ""

            return (gr.Button.update(visible=has_choices), gr.Textbox.update(visible=not has_choices or csv_mode, value=axis_values),
                    gr.update(choices=choices if has_choices else None, visible=has_choices and not csv_mode, value=axis_values_dropdown))

        x_type.change(fn=select_axis, inputs=[x_type, x_values, x_values_dropdown, csv_mode], outputs=[fill_x_button, x_values, x_values_dropdown])
        y_type.change(fn=select_axis, inputs=[y_type, y_values, y_values_dropdown, csv_mode], outputs=[fill_y_button, y_values, y_values_dropdown])
        z_type.change(fn=select_axis, inputs=[z_type, z_values, z_values_dropdown, csv_mode], outputs=[fill_z_button, z_values, z_values_dropdown])

        def change_choice_mode(csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown):
            """ "Use text inputs" チェックボックスが変更されたときの処理 """
            _fill_x_button, _x_values, _x_values_dropdown = select_axis(x_type, x_values, x_values_dropdown, csv_mode)
            _fill_y_button, _y_values, _y_values_dropdown = select_axis(y_type, y_values, y_values_dropdown, csv_mode)
            _fill_z_button, _z_values, _z_values_dropdown = select_axis(z_type, z_values, z_values_dropdown, csv_mode)
            return _fill_x_button, _x_values, _x_values_dropdown, _fill_y_button, _y_values, _y_values_dropdown, _fill_z_button, _z_values, _z_values_dropdown

        csv_mode.change(fn=change_choice_mode, inputs=[csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown], outputs=[fill_x_button, x_values, x_values_dropdown, fill_y_button, y_values, y_values_dropdown, fill_z_button, z_values, z_values_dropdown])

        def get_dropdown_update_from_params(axis, params):
            """ PNG Infoからパラメータを読み込むときの処理 """
            val_key = f"{axis} Values"
            vals = params.get(val_key, "")
            valslist = csv_string_to_list_strip(vals)
            return gr.update(value=valslist)

        # PNG Infoに値を書き込むための設定
        self.infotext_fields = (
            (x_type, "X Type"),
            (x_values, "X Values"),
            (x_values_dropdown, lambda params: get_dropdown_update_from_params("X", params)),
            (y_type, "Y Type"),
            (y_values, "Y Values"),
            (y_values_dropdown, lambda params: get_dropdown_update_from_params("Y", params)),
            (z_type, "Z Type"),
            (z_values, "Z Values"),
            (z_values_dropdown, lambda params: get_dropdown_update_from_params("Z", params)),
        )

        # UIコンポーネントのリストを返す (runメソッドの引数として渡される)
        return [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode, draw_grid, split_grids_by_batch, draw_legend_x, draw_legend_y, draw_legend_z]

    def run(self, p, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode, draw_grid, split_grids_by_batch, draw_legend_x, draw_legend_y, draw_legend_z):
        """
        「Generate」ボタンが押されたときに実行されるメインの処理。
        p: 画像生成の全パラメータを持つオブジェクト (StableDiffusionProcessing)
        ...: ui()メソッドが返したUIコンポーネントの値が、順番に引数として渡される
        """
        
        # 軸が選択されていない場合は 0 (Nothing) にフォールバック
        x_type, y_type, z_type = x_type or 0, y_type or 0, z_type or 0

        # シードを固定 (Keep -1 for seeds がオフの場合)
        if not no_fixed_seeds:
            modules.processing.fix_seed(p)

        # "Save grid" がオフの場合、バッチサイズを強制的に1にする (オリジナルの動作)
        if not opts.return_grid:
            p.batch_size = 1

        def process_axis(opt, vals, vals_dropdown):
            """
            UIから入力された軸の値を解釈し、処理用の値のリストに変換する。
            (例: "1-5" を [1, 2, 3, 4, 5] に、"A, B" を ["A", "B"] に変換)
            """
            if opt.label == 'Nothing':
                return [0]

            # ドロップダウンとテキストボックスのどちらから値を取得するかを決定
            if opt.choices is not None and not csv_mode:
                valslist = vals_dropdown
            elif opt.prepare is not None:
                valslist = opt.prepare(vals)
            else:
                valslist = csv_string_to_list_strip(vals)

            # --- 範囲指定 ("1-5", "1-10(2)", "1-10[5]") の解釈 ---
            if opt.type == int:
                valslist_ext = []
                for val in valslist:
                    if val.strip() == '': continue
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1)); end = int(m.group(2)) + 1; step = int(m.group(3)) if m.group(3) is not None else 1
                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1)); end = int(mc.group(2)); num = int(mc.group(3)) if mc.group(3) is not None else 1
                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else:
                        valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []
                for val in valslist:
                    if val.strip() == '': continue
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1)); end = float(m.group(2)); step = float(m.group(3)) if m.group(3) is not None else 1
                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1)); end = float(mc.group(2)); num = int(mc.group(3)) if mc.group(3) is not None else 1
                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))
            # --- 範囲指定の解釈ここまで ---

            # 値を正しい型に変換
            valslist = [opt.type(x) for x in valslist]
            
            # 軸ごとの確認関数を実行 (例: サンプラー名が実在するか)
            if opt.confirm:
                opt.confirm(p, valslist)
            return valslist

        # X, Y, Z軸それぞれについて、値のリストを作成
        x_opt = self.current_axis_options[x_type]
        if x_opt.choices is not None and not csv_mode:
            x_values = list_to_csv_string(x_values_dropdown)
        xs = process_axis(x_opt, x_values, x_values_dropdown)

        y_opt = self.current_axis_options[y_type]
        if y_opt.choices is not None and not csv_mode:
            y_values = list_to_csv_string(y_values_dropdown)
        ys = process_axis(y_opt, y_values, y_values_dropdown)

        z_opt = self.current_axis_options[z_type]
        if z_opt.choices is not None and not csv_mode:
            z_values = list_to_csv_string(z_values_dropdown)
        zs = process_axis(z_opt, z_values, z_values_dropdown)

        # グリッドが巨大になりすぎるのを防ぐ
        Image.MAX_IMAGE_PIXELS = None
        grid_mp = round(len(xs) * len(ys) * len(zs) * p.width * p.height / 1000000)
        assert grid_mp < opts.img_max_size_mp, f'Error: Resulting grid would be too large ({grid_mp} MPixels) (max configured size is {opts.img_max_size_mp} MPixels)'

        def fix_axis_seeds(axis_opt, axis_list):
            """ 軸がSeedの場合、-1をランダムなシード値に置き換える """
            if axis_opt.label in ['Seed', 'Var. seed']:
                return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)
            zs = fix_axis_seeds(z_opt, zs)

        # UIに進捗バーの総ステップ数を計算して設定
        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys) * len(zs)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs) * len(zs)
        elif z_opt.label == 'Steps':
            total_steps = sum(zs) * len(xs) * len(ys)
        else:
            total_steps = p.steps * len(xs) * len(ys) * len(zs)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            if x_opt.label == "Hires steps": total_steps += sum(xs) * len(ys) * len(zs)
            elif y_opt.label == "Hires steps": total_steps += sum(ys) * len(xs) * len(zs)
            elif z_opt.label == "Hires steps": total_steps += sum(zs) * len(xs) * len(ys)
            elif p.hr_second_pass_steps: total_steps += p.hr_second_pass_steps * len(xs) * len(ys) * len(zs)
            else: total_steps *= 2

        total_steps *= p.n_iter

        image_cell_count = p.n_iter * p.batch_size
        cell_console_text = f"; {image_cell_count} images per cell" if image_cell_count > 1 else ""
        plural_s = 's' if len(zs) > 1 else ''
        print(f"X/Y/Z plot will create {len(xs) * len(ys) * len(zs) * image_cell_count} images on {len(zs)} {len(xs)}x{len(ys)} grid{plural_s}{cell_console_text}. (Total steps to process: {total_steps})")
        shared.total_tqdm.updateTotal(total_steps)

        # 他のスクリプトから参照できるように、現在の軸情報をstateに保存
        state.xyz_plot_x = AxisInfo(x_opt, xs)
        state.xyz_plot_y = AxisInfo(y_opt, ys)
        state.xyz_plot_z = AxisInfo(z_opt, zs)

        # 軸のコスト（モデル読み込みなど）に基づいて、ループの順序を最適化
        first_axes_processed = 'z'
        second_axes_processed = 'y'
        if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
            first_axes_processed = 'x'
            if y_opt.cost > z_opt.cost: second_axes_processed = 'y'
            else: second_axes_processed = 'z'
        elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
            first_axes_processed = 'y'
            if x_opt.cost > z_opt.cost: second_axes_processed = 'x'
            else: second_axes_processed = 'z'
        elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
            first_axes_processed = 'z'
            if x_opt.cost > y_opt.cost: second_axes_processed = 'x'
            else: second_axes_processed = 'y'

        # グリッド画像のメタデータ（infotext）を格納するリスト
        grid_infotext = [None] * (1 + len(zs))

        def cell(x, y, z, ix, iy, iz):
            """
            1セル分の画像生成を実行する内部関数。
            `draw_xyz_grid` または `draw_xyz_grid_split_by_batch` から呼び出される。
            """
            if shared.state.interrupted or state.stopping_generation: return Processed(p, [], p.seed, "")

            # pをコピーして、現在の軸の値(x, y, z)を適用
            pc = copy(p)
            pc.styles = pc.styles[:]
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)
            z_opt.apply(pc, z, zs)

            # "Vary seeds" が有効な場合、シード値を変更
            xdim = len(xs) if vary_seeds_x else 1
            ydim = len(ys) if vary_seeds_y else 1
            if vary_seeds_x: pc.seed += ix
            if vary_seeds_y: pc.seed += iy * xdim
            if vary_seeds_z: pc.seed += iz * xdim * ydim

            try:
                # 実際に画像生成を実行
                res = process_images(pc)
            except Exception as e:
                errors.display(e, "generating image for xyz plot")
                res = Processed(p, [], p.seed, "")

            # --- グリッド用のメタデータ（infotext）を生成 ---
            # 各サブグリッドの最初のセル(ix=0, iy=0)が呼ばれたときに、そのグリッドのメタデータを生成
            subgrid_index = 1 + iz
            if grid_infotext[subgrid_index] is None and ix == 0 and iy == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                pc.extra_generation_params['Script'] = self.title()
                if x_opt.label != 'Nothing':
                    pc.extra_generation_params["X Type"] = x_opt.label
                    pc.extra_generation_params["X Values"] = x_values
                    if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds: pc.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs])
                if y_opt.label != 'Nothing':
                    pc.extra_generation_params["Y Type"] = y_opt.label
                    pc.extra_generation_params["Y Values"] = y_values
                    if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds: pc.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys])
                grid_infotext[subgrid_index] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            
            # 最初のセル(ix=0, iy=0, iz=0)が呼ばれたときに、Z軸グリッド（メイングリッド）のメタデータを生成
            if grid_infotext[0] is None and ix == 0 and iy == 0 and iz == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                if z_opt.label != 'Nothing':
                    pc.extra_generation_params["Z Type"] = z_opt.label
                    pc.extra_generation_params["Z Values"] = z_values
                    if z_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds: pc.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs])
                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            
            # 生成結果 (Processed オブジェクト) を返す
            return res

        # --- メインロジック ---
        with SharedSettingsStackHelper():
            
            # --- ▼▼▼ 新機能の分岐 ▼▼▼ ---
            # "Split grids by batch size" がオン、かつ バッチサイズ > 1 の場合
            if split_grids_by_batch and p.batch_size > 1:
                
                # コア機能によるセルごとの自動グリッド生成を一時的に無効化
                # (個別画像のみを収集するため)
                p.override_settings['grid_save'] = False
                
                # 新しい描画関数を呼び出す
                processed = draw_xyz_grid_split_by_batch(
                    p, 
                    xs=xs, ys=ys, zs=zs, 
                    x_labels=[x_opt.format_value(p, x_opt, x) for x in xs], 
                    y_labels=[y_opt.format_value(p, y_opt, y) for y in ys], 
                    z_labels=[z_opt.format_value(p, z_opt, z) for z in zs], 
                    cell=cell, 
                    margin_size=margin_size, 
                    draw_legend_x=draw_legend_x, 
                    draw_legend_y=draw_legend_y, 
                    draw_legend_z=draw_legend_z, 
                    include_sub_grids=include_sub_grids
                )
                
                # 設定を元に戻す
                p.override_settings.pop('grid_save', None)

                if not processed.images: return processed
                
                # --- 新機能用の画像保存ループ ---
                if opts.grid_save:
                    # メタデータ(all_prompts, all_seeds)は代表値しか含まないため、リストの長さを取得
                    num_prompts = len(processed.all_prompts)
                    num_seeds = len(processed.all_seeds)

                    # Zグリッド + XYサブグリッド のすべてをループして保存
                    for i, grid_image in enumerate(processed.images):
                        # 【エラー修正】画像数(i)がメタデータ数を超えても、剰余(%)でインデックスをループさせ、エラーを防ぐ
                        prompt_index = i % num_prompts if num_prompts > 0 else 0
                        seed_index = i % num_seeds if num_seeds > 0 else 0
                        
                        images.save_image(
                            grid_image, 
                            p.outpath_grids, 
                            "xyz_grid", 
                            info=processed.infotexts[i], # infotextsは画像ごとに固有のものが正しく入っている
                            extension=opts.grid_format, 
                            prompt=processed.all_prompts[prompt_index], 
                            seed=processed.all_seeds[seed_index], 
                            grid=True, 
                            p=p
                        )
                
                # 新機能モードはここで処理を終了し、結果を返す
                return processed
            
            # --- ▼▼▼ オリジナルのロジック ▼▼▼ ---
            else:
                # "Split grids" がオフの場合、オリジナルの描画関数を呼び出す
                
                # 新しい軸ごと凡例チェックボックスを、従来の単一の凡例フラグに変換
                draw_legend_for_original = draw_legend_x or draw_legend_y or draw_legend_z
                
                processed = draw_xyz_grid(
                    p, 
                    xs=xs, ys=ys, zs=zs, 
                    x_labels=[x_opt.format_value(p, x_opt, x) for x in xs], 
                    y_labels=[y_opt.format_value(p, y_opt, y) for y in ys], 
                    z_labels=[z_opt.format_value(p, z_opt, z) for z in zs], 
                    cell=cell, 
                    draw_legend=draw_legend_for_original, # 従来のフラグを渡す
                    include_lone_images=include_lone_images, 
                    include_sub_grids=include_sub_grids, 
                    first_axes_processed=first_axes_processed, 
                    second_axes_processed=second_axes_processed, 
                    margin_size=margin_size, 
                    draw_grid=draw_grid
                )

        # --- オリジナルのロジックの後処理 ---
        
        if not processed.images:
            return processed

        z_count = len(zs)
        
        # グリッドのメタデータを、生成されたメタデータで上書き
        if draw_grid:
            processed.infotexts[:1 + z_count] = grid_infotext[:1 + z_count]

        # "Include Sub Images" がオフの場合、個別の画像を結果から削除
        if not include_lone_images:
            processed.images = processed.images[:z_count + 1] if draw_grid else []

        # "Save grid" がオンの場合、グリッド画像を保存
        if draw_grid and opts.grid_save:
            grid_count = z_count + 1 if z_count > 1 else 1
            for g in range(grid_count):
                adj_g = g - 1 if g > 0 else g
                images.save_image(processed.images[g], p.outpath_grids, "xyz_grid", info=processed.infotexts[g], extension=opts.grid_format, prompt=processed.all_prompts[adj_g], seed=processed.all_seeds[adj_g], grid=True, p=processed)
                if not include_sub_grids:  # "Include Sub Grids" がオフなら、Zグリッド(0番目)のみ保存して終了
                    break

        # "Include Sub Grids" がオフの場合、サブグリッドの情報を結果から削除
        if draw_grid and not include_sub_grids:
            if z_count > 1:
                for _ in range(z_count):
                    if len(processed.images) > 1: del processed.images[1]
                    if len(processed.all_prompts) > 1: del processed.all_prompts[1]
                    if len(processed.all_seeds) > 1: del processed.all_seeds[1]
                    if len(processed.infotexts) > 1: del processed.infotexts[1]
        
        return processed