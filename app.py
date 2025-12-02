import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison
import utils
import pandas as pd
import cv2
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas

# Page Config
st.set_page_config(
    page_title="IQA Visualization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <a href="https://github.com/XuBao12/IQA-Visualization" target="_blank" class="github-corner" aria-label="View source on GitHub">
        <svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: absolute; top: 0; border: 0; right: 0; z-index: 9999;" aria-hidden="true">
            <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
            <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
            <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
        </svg>
    </a>
    <style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 IQA Visualization")
st.markdown(
    "图像超分质量评价工作台：上传 GT 和 SR 图像，一键计算 PSNR/SSIM/LPIPS 等指标并进行可视化对比。"
)

# --- Sidebar ---
with st.sidebar:
    st.header("1. 设置")

    # Metrics Selection
    available_metrics = [
        "PSNR",
        "SSIM",
        "LPIPS",
        "FID",
        "CLIPIQA",
        "CNNIQA",
        "MUSIQ",
        "DISTS",
        "SRE",
        "GSD",
    ]
    selected_metrics = st.multiselect(
        "选择指标 (Select Metrics)",
        available_metrics,
        default=["PSNR", "SSIM", "LPIPS"],
    )

    lpips_net = st.selectbox("LPIPS 主干网络 (Backbone)", ["vgg", "alex"], index=0)
    crop_border = st.number_input(
        "边缘裁剪 (Crop Border px)",
        min_value=0,
        value=4,
        help="SR 常用评估设置，切除边缘像素",
    )
    use_y_channel = st.checkbox(
        "转换为 Y 通道计算 PSNR/SSIM",
        value=True,
        help="SR 论文通常在 Y 通道计算指标",
    )

    st.header("2. 图像输入")
    st.caption("支持输入本地路径或者服务器6上路径")
    input_mode = st.radio(
        "输入模式",
        [
            "本地路径 单图输入",
            "本地路径 文件夹输入",
            "服务器路径 单图输入",
            "服务器路径 文件夹输入",
        ],
        index=0,
        help="支持本地输入或服务器路径输入，文件夹大于1GB建议从服务器输入",
    )

    gt_file = None
    sr_file = None
    gt_path = None
    sr_path = None
    gt_map = {}
    sr_map = {}
    valid_files = []

    # Initialize session state for folder navigation
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if input_mode == "本地路径 单图输入":
        gt_file = st.file_uploader(
            "上传参考图 (GT)", type=["png", "jpg", "jpeg", "bmp", "tiff"]
        )
        sr_file = st.file_uploader(
            "上传失真图 (SR)", type=["png", "jpg", "jpeg", "bmp", "tiff"]
        )
    elif input_mode == "服务器路径 单图输入":
        gt_path = st.text_input("GT 图像路径", placeholder="/path/to/gt.png")
        sr_path = st.text_input("SR 图像路径", placeholder="/path/to/sr.png")
    elif input_mode == "服务器路径 文件夹输入":
        gt_folder = st.text_input("GT 文件夹路径", placeholder="/path/to/gt_folder")
        sr_folder = st.text_input("SR 文件夹路径", placeholder="/path/to/sr_folder")

        if gt_folder and sr_folder:
            if os.path.isdir(gt_folder) and os.path.isdir(sr_folder):
                # Get file lists
                gt_files = utils.get_image_files(gt_folder)

                # Filter files that exist in both folders (assuming same filename)
                valid_files = [
                    f for f in gt_files if os.path.exists(os.path.join(sr_folder, f))
                ]

                if not valid_files:
                    st.error("在两个文件夹中未找到匹配的图像文件（文件名必须相同）。")
                else:
                    st.sidebar.markdown(f"**找到 {len(valid_files)} 张匹配图像**")

                    # Ensure index is valid
                    if st.session_state.current_index >= len(valid_files):
                        st.session_state.current_index = 0

                    # Jump to index
                    jump_to = st.sidebar.number_input(
                        "跳转到图片",
                        min_value=1,
                        max_value=len(valid_files),
                        value=st.session_state.current_index + 1,
                        key="jump_to_server",
                    )
                    st.session_state.current_index = jump_to - 1

                    # Navigation Buttons
                    col_prev, col_next = st.sidebar.columns(2)
                    if col_prev.button("⬅️ 上一张"):
                        st.session_state.current_index = max(
                            0, st.session_state.current_index - 1
                        )
                    if col_next.button("下一张 ➡️"):
                        st.session_state.current_index = min(
                            len(valid_files) - 1, st.session_state.current_index + 1
                        )

                    # Display current file info
                    current_file = valid_files[st.session_state.current_index]
                    st.sidebar.info(
                        f"当前文件: `{current_file}`\n({st.session_state.current_index + 1}/{len(valid_files)})"
                    )

                    # Set paths for loading
                    gt_path = os.path.join(gt_folder, current_file)
                    sr_path = os.path.join(sr_folder, current_file)
            else:
                st.error("无效的文件夹路径。")

    elif input_mode == "本地路径 文件夹输入":
        st.info(
            "💡 提示：Streamlit 不支持直接选择文件夹。请点击下方按钮，进入文件夹后按 `Ctrl+A` 全选所有图片进行上传。"
        )
        gt_files_upload = st.file_uploader(
            "上传参考图文件夹 (GT) - 请全选图片",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="请进入文件夹，按 Ctrl+A 全选所有图片上传",
        )
        sr_files_upload = st.file_uploader(
            "上传失真图文件夹 (SR) - 请全选图片",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="请进入文件夹，按 Ctrl+A 全选所有图片上传",
        )

        if gt_files_upload and sr_files_upload:
            # Create maps
            gt_map = {f.name: f for f in gt_files_upload}
            sr_map = {f.name: f for f in sr_files_upload}

            # Find intersection
            valid_files = sorted(list(set(gt_map.keys()) & set(sr_map.keys())))

            if not valid_files:
                st.error("在上传的文件中未找到匹配的图像文件（文件名必须相同）。")
            else:
                st.sidebar.markdown(f"**找到 {len(valid_files)} 张匹配图像**")

                # Ensure index is valid
                if st.session_state.current_index >= len(valid_files):
                    st.session_state.current_index = 0

                # Jump to index
                jump_to = st.sidebar.number_input(
                    "跳转到图片",
                    min_value=1,
                    max_value=len(valid_files),
                    value=st.session_state.current_index + 1,
                    key="jump_to_local",
                )
                st.session_state.current_index = jump_to - 1

                # Navigation Buttons
                col_prev, col_next = st.sidebar.columns(2)
                if col_prev.button("⬅️ 上一张"):
                    st.session_state.current_index = max(
                        0, st.session_state.current_index - 1
                    )
                if col_next.button("下一张 ➡️"):
                    st.session_state.current_index = min(
                        len(valid_files) - 1, st.session_state.current_index + 1
                    )

                # Display current file info
                current_file = valid_files[st.session_state.current_index]
                st.sidebar.info(
                    f"当前文件: `{current_file}`\n({st.session_state.current_index + 1}/{len(valid_files)})"
                )

# --- Main Content ---
img_gt_raw = None
img_sr_raw = None

try:
    if input_mode == "本地路径 单图输入":
        if gt_file and sr_file:
            img_gt_raw = utils.load_image(gt_file)
            img_sr_raw = utils.load_image(sr_file)
    elif input_mode == "服务器路径 单图输入" or input_mode == "服务器路径 文件夹输入":
        if gt_path and sr_path:
            img_gt_raw = utils.load_image_from_path(gt_path)
            img_sr_raw = utils.load_image_from_path(sr_path)
    elif input_mode == "本地路径 文件夹输入":
        if valid_files:
            current_file = valid_files[st.session_state.current_index]
            f_gt = gt_map[current_file]
            f_sr = sr_map[current_file]
            f_gt.seek(0)
            f_sr.seek(0)
            img_gt_raw = utils.load_image(f_gt)
            img_sr_raw = utils.load_image(f_sr)
except Exception as e:
    st.error(f"加载图像出错: {e}")

if img_gt_raw is not None and img_sr_raw is not None:
    # Preprocess (Resize & Crop)
    img_gt, img_sr = utils.preprocess_images(
        img_gt_raw, img_sr_raw, crop_border=crop_border
    )

    # --- Metrics Dashboard ---
    st.subheader("📊 单图评估指标")

    with st.spinner("正在计算指标..."):
        metrics = utils.calculate_metrics(
            img_gt,
            img_sr,
            use_y_channel=use_y_channel,
            lpips_net=lpips_net,
            selected_metrics=selected_metrics,
        )

    # Dynamic display of metrics
    if metrics:
        # Sort metrics based on selection order
        ordered_metrics = {k: metrics[k] for k in selected_metrics if k in metrics}

        cols = st.columns(len(ordered_metrics))
        for col, (name, value) in zip(cols, ordered_metrics.items()):
            # Determine delta color (LPIPS, FID, DISTS are lower is better)
            lower_is_better = name in ["LPIPS", "FID", "DISTS"]
            delta_color = "inverse" if lower_is_better else "normal"

            # Format value
            if np.isnan(value):
                display_val = "N/A"
            else:
                display_val = f"{value:.4f}"
                if name == "PSNR":
                    display_val += " dB"

            col.metric(name, display_val, delta_color=delta_color)
    else:
        st.warning("未选择指标。")

    # Export Data
    metrics_df = pd.DataFrame([metrics])
    # Reorder columns for CSV
    if metrics:
        cols_order = [c for c in selected_metrics if c in metrics_df.columns]
        metrics_df = metrics_df[cols_order]

    col_dl, col_cp, _ = st.columns([1, 1.5, 5], gap="small")

    with col_dl:
        st.download_button(
            label="下载 CSV",
            data=metrics_df.to_csv(index=False),
            file_name="metrics.csv",
            mime="text/csv",
        )

    # Generate HTML table (No Header)
    html_table_single = metrics_df.to_html(
        index=False, header=False, float_format="%.4f", border=1
    )

    import streamlit.components.v1 as components

    with col_cp:
        components.html(
            f"""
            <style>
                body {{ margin: 0; font-family: "Source Sans Pro", sans-serif; }}
                .btn {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 400;
                    padding: 0.25rem 0.75rem;
                    border-radius: 0.5rem;
                    min-height: 38.4px;
                    margin: 0px;
                    line-height: 1.6;
                    color: rgb(49, 51, 63);
                    background-color: rgb(255, 255, 255);
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    font-size: 1rem;
                    cursor: pointer;
                    gap: 8px;
                }}
                .btn:hover {{
                    border-color: rgb(255, 75, 75);
                    color: rgb(255, 75, 75);
                }}
                .btn:active {{
                    background-color: rgb(255, 75, 75);
                    color: white;
                }}
            </style>
            <script>
                function copyTableSingle() {{
                    const table = document.getElementById('data-table-single');
                    const range = document.createRange();
                    range.selectNode(table);
                    window.getSelection().removeAllRanges();
                    window.getSelection().addRange(range);
                    try {{
                        document.execCommand('copy');
                        const btn = document.getElementById('copy-btn-single');
                        btn.innerHTML = '✅ 已复制！';
                        setTimeout(() => {{ btn.innerHTML = '复制结果'; }}, 2000);
                    }} catch (err) {{
                        alert('复制失败');
                    }}
                    window.getSelection().removeAllRanges();
                }}
            </script>
            <div style="display: flex; align-items: center;">
                <button id="copy-btn-single" class="btn" onclick="copyTableSingle()">复制结果</button>
                <div id="data-table-single" style="position: absolute; left: -9999px;">
                    {html_table_single}
                </div>
            </div>
            """,
            height=45,
        )

    # --- Batch Evaluation (Server Folder only) ---
    is_batch_mode = (
        input_mode == "服务器路径 文件夹输入" or input_mode == "本地路径 文件夹输入"
    ) and valid_files

    if is_batch_mode:
        st.divider()
        st.subheader("📚 批量评估")

        if st.button("开始计算平均指标"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_metrics = []

            # Separate FID from per-image metrics
            per_image_metrics_selection = [m for m in selected_metrics if m != "FID"]
            calc_fid = "FID" in selected_metrics
            calc_gsd = "GSD" in selected_metrics

            total_files = len(valid_files)
            for i, filename in enumerate(valid_files):
                status_text.text(f"正在处理 {i+1}/{total_files}: {filename}")

                try:
                    if input_mode == "服务器路径 文件夹输入":
                        f_gt = os.path.join(gt_folder, filename)
                        f_sr = os.path.join(sr_folder, filename)
                        i_gt_raw = utils.load_image_from_path(f_gt)
                        i_sr_raw = utils.load_image_from_path(f_sr)
                    else:  # Local Upload
                        f_gt = gt_map[filename]
                        f_sr = sr_map[filename]
                        f_gt.seek(0)
                        f_sr.seek(0)
                        i_gt_raw = utils.load_image(f_gt)
                        i_sr_raw = utils.load_image(f_sr)

                    if i_gt_raw is not None and i_sr_raw is not None:
                        i_gt_p, i_sr_p = utils.preprocess_images(
                            i_gt_raw, i_sr_raw, crop_border=crop_border
                        )

                        if per_image_metrics_selection:
                            m = utils.calculate_metrics(
                                i_gt_p,
                                i_sr_p,
                                use_y_channel=use_y_channel,
                                lpips_net=lpips_net,
                                selected_metrics=per_image_metrics_selection,
                            )
                            m["Filename"] = filename
                            all_metrics.append(m)
                        else:
                            all_metrics.append({"Filename": filename})
                except Exception as e:
                    st.warning(f"处理失败 {filename}: {e}")

                progress_bar.progress((i + 1) / total_files)

            # Calculate FID globally
            fid_score = None
            if calc_fid:
                if input_mode == "服务器路径 文件夹输入":
                    status_text.text("正在计算 FID... (这可能需要一些时间)")
                    with st.spinner("正在计算 FID..."):
                        fid_score = utils.calculate_fid_folder(gt_folder, sr_folder)
                else:
                    st.warning(
                        "注意：本地上传模式暂不支持计算文件夹级 FID (需要物理路径)。"
                    )
            if calc_gsd:
                if input_mode == "服务器路径 文件夹输入":
                    f_gt = [os.path.join(gt_folder, filename) for filename in valid_files]
                    f_sr = [os.path.join(sr_folder, filename) for filename in valid_files]
                    status_text.text("正在计算 GSD... (这可能需要一些时间)")
                    with st.spinner("正在计算 GSD..."):
                        fid_score = utils.calculate_gsd(f_gt, f_sr)
                else:
                    st.warning(
                        "注意：本地上传模式暂不支持计算文件夹级 GSD (需要物理路径)。"
                    )


            status_text.empty()

            if all_metrics:
                df_all = pd.DataFrame(all_metrics)

                # Move Filename to first column and reorder metrics
                cols = ["Filename"]
                # Add metrics in selection order
                cols += [c for c in selected_metrics if c in df_all.columns]
                # Add any remaining columns (like FID if calculated separately but not in selection list logic)
                cols += [c for c in df_all.columns if c not in cols]

                df_all = df_all[cols]
                df_all.index = range(1, len(df_all) + 1)

                st.session_state["batch_results"] = df_all
                st.session_state["batch_fid"] = fid_score
                st.success("批量评估完成！")
            else:
                st.error("未计算任何指标。")

        if "batch_results" in st.session_state:
            df_all = st.session_state["batch_results"]
            fid_score = st.session_state.get("batch_fid", None)

            # Average
            numeric_cols = df_all.select_dtypes(include=[np.number]).columns
            avg_metrics = df_all[numeric_cols].mean()

            if fid_score is not None:
                avg_metrics["FID"] = fid_score

            st.write("### 平均指标")
            if not avg_metrics.empty:
                # Sort average metrics based on selection order
                ordered_avg_metrics = {
                    k: avg_metrics[k] for k in selected_metrics if k in avg_metrics
                }
                # Add any remaining metrics (like FID)
                for k, v in avg_metrics.items():
                    if k not in ordered_avg_metrics:
                        ordered_avg_metrics[k] = v

                cols_avg = st.columns(len(ordered_avg_metrics))
                for col, (name, value) in zip(cols_avg, ordered_avg_metrics.items()):
                    lower_is_better = name in ["LPIPS", "FID", "DISTS"]
                    delta_color = "inverse" if lower_is_better else "normal"

                    display_val = f"{value:.4f}"
                    if name == "PSNR":
                        display_val += " dB"
                    col.metric(name, display_val, delta_color=delta_color)

                # Copy friendly format
                avg_df = pd.DataFrame([ordered_avg_metrics])

                # Generate HTML table (No Header)
                html_table = avg_df.to_html(
                    index=False, header=False, float_format="%.4f", border=1
                )

            st.write("### 详细结果")
            if fid_score is not None:
                st.caption(
                    "*注意：FID 是针对整个文件夹全局计算的，不会显示在单张图片的表格中。*"
                )
            st.dataframe(df_all)

            col_dl_batch, col_cp_batch, _ = st.columns([1, 1.5, 5], gap="small")

            with col_dl_batch:
                st.download_button(
                    label="下载批量结果 CSV",
                    data=df_all.to_csv(index=False),
                    file_name="batch_metrics.csv",
                    mime="text/csv",
                )

            with col_cp_batch:
                components.html(
                    f"""
                    <style>
                        body {{ margin: 0; font-family: "Source Sans Pro", sans-serif; }}
                        .btn {{
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: 400;
                            padding: 0.25rem 0.75rem;
                            border-radius: 0.5rem;
                            min-height: 38.4px;
                            margin: 0px;
                            line-height: 1.6;
                            color: rgb(49, 51, 63);
                            background-color: rgb(255, 255, 255);
                            border: 1px solid rgba(49, 51, 63, 0.2);
                            font-size: 1rem;
                            cursor: pointer;
                            gap: 8px;
                        }}
                        .btn:hover {{
                            border-color: rgb(255, 75, 75);
                            color: rgb(255, 75, 75);
                        }}
                        .btn:active {{
                            background-color: rgb(255, 75, 75);
                            color: white;
                        }}
                    </style>
                    <script>
                        function copyTable() {{
                            const table = document.getElementById('data-table');
                            const range = document.createRange();
                            range.selectNode(table);
                            window.getSelection().removeAllRanges();
                            window.getSelection().addRange(range);
                            try {{
                                document.execCommand('copy');
                                const btn = document.getElementById('copy-btn');
                                btn.innerHTML = '✅ 已复制！';
                                setTimeout(() => {{ btn.innerHTML = '复制结果'; }}, 2000);
                            }} catch (err) {{
                                alert('复制失败');
                            }}
                            window.getSelection().removeAllRanges();
                        }}
                    </script>
                    <div style="display: flex; align-items: center;">
                        <button id="copy-btn" class="btn" onclick="copyTable()">复制结果</button>
                        <div id="data-table" style="position: absolute; left: -9999px;">
                            {html_table}
                        </div>
                    </div>
                    """,
                    height=45,
                )

    # --- Visual Comparison ---
    st.subheader("👁️ 可视化对比")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "↔️ 滑块对比",
            "🖼️ 原图对比",
            "🔥 误差热力图",
            "📈 频谱分析",
            "🧶 纹理分析",
            "📏 边缘分析",
        ]
    )

    with tab1:
        st.write("左右拖动滑块对比细节：")
        # streamlit-image-comparison expects images in RGB
        image_comparison(
            img1=img_gt,
            img2=img_sr,
            label1="参考图 (GT)",
            label2="失真图 (SR)",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )

    with tab2:
        st.write("原图对比：左侧为参考图 (GT)，右侧为失真图 (SR)。")
        col_orig1, col_orig2 = st.columns(2)
        with col_orig1:
            st.image(img_gt, caption="参考图 (GT)", use_container_width=True)
        with col_orig2:
            st.image(img_sr, caption="失真图 (SR)", use_container_width=True)

    with tab3:
        st.write("差值热力图: 颜色越亮表示误差越大。")
        error_map = utils.get_error_map(img_gt, img_sr)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(error_map, cmap="jet")
        plt.colorbar(im, ax=ax)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.write("频域分析: 检查高频信息丢失或伪影。")

        ftab1, ftab2, ftab3, ftab4 = st.tabs(
            ["2D 频谱图", "1D 功率谱", "差异曲线", "MTF 曲线"]
        )

        with ftab1:
            fft_gt = utils.get_fft_spectrum(img_gt)
            fft_sr = utils.get_fft_spectrum(img_sr)

            col_fft1, col_fft2 = st.columns(2)

            with col_fft1:
                st.caption("GT 频谱")
                fig1, ax1 = plt.subplots()
                ax1.imshow(fft_gt, cmap="gray")
                ax1.axis("off")
                st.pyplot(fig1)
                plt.close(fig1)

            with col_fft2:
                st.caption("SR 频谱")
                fig2, ax2 = plt.subplots()
                ax2.imshow(fft_sr, cmap="gray")
                ax2.axis("off")
                st.pyplot(fig2)
                plt.close(fig2)

        # Calculate 1D PSD for tab2 and tab3
        psd_gt = utils.get_1d_power_spectrum(img_gt)
        psd_sr = utils.get_1d_power_spectrum(img_sr)

        # Ensure same length for plotting
        min_len = min(len(psd_gt), len(psd_sr))
        psd_gt = psd_gt[:min_len]
        psd_sr = psd_sr[:min_len]

        with ftab2:
            st.write("#### 1D 功率谱对比")
            # Plot 1D Power Spectrum
            fig_1d, ax_1d = plt.subplots(figsize=(10, 4))
            ax_1d.plot(
                10 * np.log10(psd_gt + 1e-8), label="GT", alpha=0.8, linewidth=1.5
            )
            ax_1d.plot(
                10 * np.log10(psd_sr + 1e-8), label="SR", alpha=0.8, linewidth=1.5
            )
            ax_1d.set_xlabel("Frequency (Radius)")
            ax_1d.set_ylabel("Log Power (dB)")
            ax_1d.set_title("1D Power Spectrum")
            ax_1d.legend()
            ax_1d.grid(True, alpha=0.3)
            st.pyplot(fig_1d)
            plt.close(fig_1d)

        with ftab3:
            st.write("#### SR 与 GT 差异曲线 (Log Power Difference)")
            st.caption("值 > 0 表示 SR 在该频率分量能量高于 GT，值 < 0 表示低于 GT。")

            diff_curve = 10 * np.log10(psd_sr + 1e-8) - 10 * np.log10(psd_gt + 1e-8)

            fig_diff, ax_diff = plt.subplots(figsize=(10, 4))
            ax_diff.plot(diff_curve, label="SR - GT", color="red", linewidth=1.5)
            ax_diff.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax_diff.set_xlabel("Frequency (Radius)")
            ax_diff.set_ylabel("Difference (Log Power) (dB)")
            ax_diff.set_title("Spectrum Difference")
            ax_diff.legend()
            ax_diff.grid(True, alpha=0.3)
            st.pyplot(fig_diff)
            plt.close(fig_diff)

        with ftab4:
            st.write("#### MTF 曲线 (Modulation Transfer Function)")
            st.caption(
                "近似计算：SR 与 GT 的幅度谱之比 (Magnitude Spectrum Ratio)。值 = 1 表示完美还原，> 1 表示锐化/伪影，< 1 表示模糊。"
            )

            # Calculate MTF: sqrt(PSD_SR) / sqrt(PSD_GT)
            mtf_curve = np.sqrt(psd_sr) / (np.sqrt(psd_gt) + 1e-8)

            fig_mtf, ax_mtf = plt.subplots(figsize=(10, 4))
            ax_mtf.plot(mtf_curve, label="MTF (SR / GT)", color="purple", linewidth=1.5)
            ax_mtf.axhline(
                1, color="black", linestyle="--", alpha=0.5, label="Ideal (1.0)"
            )
            ax_mtf.set_xlabel("Frequency (Radius)")
            ax_mtf.set_ylabel("Relative Response")
            ax_mtf.set_title("Approximate MTF Curve")
            ax_mtf.legend()
            ax_mtf.grid(True, alpha=0.3)
            ax_mtf.set_ylim(0, 2.0)
            st.pyplot(fig_mtf)
            plt.close(fig_mtf)

    with tab5:
        st.write("纹理一致性分析: 使用 Gabor 滤波器组提取纹理能量和主方向。")

        with st.spinner("正在进行纹理分析..."):
            energy_gt, orient_gt = utils.get_texture_analysis(img_gt)
            energy_sr, orient_sr = utils.get_texture_analysis(img_sr)

        st.write("#### 1. 纹理能量图 (Texture Energy)")
        st.caption("颜色越亮表示该区域纹理越丰富/高频能量越强。")
        col_tex1, col_tex2 = st.columns(2)
        with col_tex1:
            st.image(energy_gt, caption="GT 纹理能量", use_container_width=True)
        with col_tex2:
            st.image(energy_sr, caption="SR 纹理能量", use_container_width=True)

        st.write("#### 2. 纹理方向图 (Texture Orientation)")
        st.caption("不同颜色代表不同的纹理主方向，亮度代表纹理强度。")
        col_ori1, col_ori2 = st.columns(2)
        with col_ori1:
            st.image(orient_gt, caption="GT 纹理方向", use_container_width=True)
        with col_ori2:
            st.image(orient_sr, caption="SR 纹理方向", use_container_width=True)

    with tab6:
        st.write("边缘分析: 对比 GT 与 SR 的边缘结构，检查是否丢失细节或产生伪影。")

        edge_method = st.radio(
            "选择边缘检测算子", ["Canny", "Sobel", "Laplacian"], horizontal=True
        )

        edge_gt = utils.get_edge_analysis(img_gt, method=edge_method)
        edge_sr = utils.get_edge_analysis(img_sr, method=edge_method)

        col_edge1, col_edge2 = st.columns(2)
        with col_edge1:
            st.image(
                edge_gt, caption=f"GT 边缘 ({edge_method})", use_container_width=True
            )
        with col_edge2:
            st.image(
                edge_sr, caption=f"SR 边缘 ({edge_method})", use_container_width=True
            )

        st.write("#### 边缘差异图 (Edge Difference)")
        st.caption("显示 SR 与 GT 边缘图的差异。")

        # Calculate difference
        edge_diff = cv2.absdiff(edge_gt, edge_sr)

        st.image(edge_diff, caption="边缘差异 (绝对差值)", clamp=True, channels="GRAY")

    # --- ROI Crop & Zoom ---
    st.subheader("✂️ 局部裁剪对比")
    st.info(
        "在下方 GT 图像上**点击并拖动鼠标**绘制矩形框，右侧将显示 GT 和 SR 的对应局部放大图。"
    )

    col_crop_main, col_crop_result = st.columns([1.5, 1])

    with col_crop_main:
        st.caption("参考图 (GT) - 在此绘制选框")

        # Prepare image for canvas
        img_gt_pil = Image.fromarray(img_gt)

        # Calculate canvas dimensions to fit layout
        canvas_width = 600
        w_orig, h_orig = img_gt_pil.size
        if w_orig > 0:
            scale_factor = canvas_width / w_orig
            canvas_height = int(h_orig * scale_factor)
        else:
            canvas_height = 400
            scale_factor = 1.0

        # Create Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=img_gt_pil,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="roi_canvas",
            display_toolbar=True,
        )

    with col_crop_result:
        st.caption("裁剪区域放大 (Zoom x4)")

        if (
            canvas_result.json_data is not None
            and len(canvas_result.json_data["objects"]) > 0
        ):
            # Get the last drawn object
            obj = canvas_result.json_data["objects"][-1]

            # Get coordinates from canvas
            left_c = int(obj["left"])
            top_c = int(obj["top"])
            width_c = int(obj["width"])
            height_c = int(obj["height"])

            # Map back to original image coordinates
            left = int(left_c / scale_factor)
            top = int(top_c / scale_factor)
            width = int(width_c / scale_factor)
            height = int(height_c / scale_factor)

            # Boundary checks
            left = max(0, min(left, w_orig - 1))
            top = max(0, min(top, h_orig - 1))
            width = max(1, min(width, w_orig - left))
            height = max(1, min(height, h_orig - top))

            if width > 0 and height > 0:
                # Crop GT
                patch_gt = img_gt[top : top + height, left : left + width]
                # Crop SR (same coordinates)
                patch_sr = img_sr[top : top + height, left : left + width]

                # Zoom
                zoom_factor = 4
                h_patch, w_patch, _ = patch_gt.shape

                # Prevent empty patch
                if h_patch > 0 and w_patch > 0:
                    patch_gt_zoom = cv2.resize(
                        patch_gt,
                        (w_patch * zoom_factor, h_patch * zoom_factor),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    patch_sr_zoom = cv2.resize(
                        patch_sr,
                        (w_patch * zoom_factor, h_patch * zoom_factor),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    st.image(patch_gt_zoom, caption="GT 局部", use_container_width=True)
                    st.image(patch_sr_zoom, caption="SR 局部", use_container_width=True)
                else:
                    st.warning("选择的区域太小。")
            else:
                st.info("请选择一个区域。")
        else:
            st.info("👈 请在左侧图像上绘制一个框。")

else:
    st.info("👈 请在左侧侧边栏上传 GT 和 SR 图像以开始分析。")
