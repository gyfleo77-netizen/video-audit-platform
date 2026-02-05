#!/bin/bash

# =================================================
# === 视频分析服务器 - 启动与清理脚本 (最终版) ===
# =================================================

# 定义前端和后端服务使用的端口
FRONTEND_PORT=8000
BACKEND_PORT=5000 # 确保此端口与 appp_api.py 中 Flask 设置的端口一致
FRONTEND_URL="http://localhost:$FRONTEND_PORT"

# 不论脚本从哪里执行，都切换到脚本文件所在的目录
cd "$(dirname "$0")"

#
# 1. 强制清理旧进程 (核心步骤)
#
echo "--> 正在强制清理可能残留的旧服务进程..."
# 使用 pkill 命令通过进程名模糊匹配并杀死进程，-f 参数表示匹配完整命令行
# 这是最有效的方式，可以杀死僵尸进程
pkill -9 -f "python appp_api.py"
pkill -9 -f "python3 -m http.server"
echo "--> 清理完成。等待2秒确保系统资源已释放..."
sleep 2

#
# 2. 激活虚拟环境
#
echo "--> 正在激活 Python 虚拟环境: ./venv"
source ./venv/bin/activate

#
# 3. 设置必要的库路径
#
#echo "--> 正在定位并设置 CUDA/cuDNN 库路径..."
#CUDNN_LIB_DIR="$(find "$VIRTUAL_ENV" -type f -name 'libcudnn*.so.9*' | head -n1 | xargs -r dirname)"
#TORCH_LIB_DIR="$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
#export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$TORCH_LIB_DIR"
#echo "[信息] LD_LIBRARY_PATH 已设置为: $LD_LIBRARY_PATH"

# 如果您想查看具体的路径，可以取消下面这行的注释
# echo "$LD_LIBRARY_PATH"

# 3. 设置所有必要的动态库路径 (核心步骤)
#
echo "--> 正在定位并设置所有 CUDA/cuDNN/Torch 相关的库路径..."

# 在虚拟环境的 site-packages 目录下，查找所有包含 .so 文件的目录
# 并筛选出与 nvidia, torch, ctranslate2 相关的
ALL_LIBS_DIRS=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/" -name '*.so*' -exec dirname {} \; | grep -E 'nvidia|torch|ctranslate2' | sort -u | tr '\n' ':')

# 将找到的所有目录，以最高优先级设置到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${ALL_LIBS_DIRS}:${LD_LIBRARY_PATH}"

echo "[信息] LD_LIBRARY_PATH 已被全面设置为，确保所有GPU库可见。"
# 如果您想查看具体的路径，可以取消下面这行的注释
# echo "$LD_LIBRARY_PATH"

#
# 4. 启动所有服务 (后端 + 前端)
#
echo "--> 一切就绪，正在启动所有服务..."
echo "    启动 Flask 后端 API (后台运行)..."
python appp_api.py &
BACKEND_PID=$! # 记录后端进程ID

echo "    启动 HTTP 前端文件服务器 (后台运行)..."
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$! # 记录前端进程ID

echo "[信息] 后端服务 PID: $BACKEND_PID | 前端服务 PID: $FRONTEND_PID"

#
# 5. 等待服务启动并自动打开浏览器
#
echo "--> 等待 3 秒让服务充分启动..."
sleep 3

echo "--> 服务器已在后台运行，正在浏览器中打开前端页面..."
echo "    访问地址: $FRONTEND_URL"
xdg-open "$FRONTEND_URL"

echo ""
echo "================================================="
echo "=== 启动完成！请在浏览器窗口中进行操作。 ==="
echo "=== 所有服务都在后台运行中。"
echo "=== 要停止所有服务, 请关闭此终端或按 Ctrl+C"
echo "================================================="

# 定义一个函数，用于在脚本退出时杀死所有子进程
cleanup() {
    echo ""
    echo "--> 收到退出信号，正在停止所有后台服务..."
    # 使用 kill 命令并传入进程ID，确保只杀死由本脚本启动的进程
    kill -9 $BACKEND_PID $FRONTEND_PID
    echo "--> 所有服务已停止。"
    exit 0
}

# 使用 trap 捕获退出信号 (Ctrl+C, 终端关闭等)，然后执行 cleanup 函数
trap cleanup SIGINT SIGTERM

# 等待后台进程，使脚本保持运行状态，以便 trap 能够工作
# 这样按 Ctrl+C 才能被捕获
wait