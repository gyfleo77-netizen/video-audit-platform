import os
import tempfile
from datetime import datetime

import yt_dlp
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from .config import DOWNLOAD_FOLDER, SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from .runtime import determine_violation_status, run_full_analysis

app = Flask(__name__)
CORS(app)  # 允许跨域请求，让前端可以访问

app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS

db = SQLAlchemy(app)

class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 每条记录的唯一ID
    user_ip = db.Column(db.String(45), nullable=False)  # 记录用户IP
    video_filename = db.Column(db.String(255), nullable=False) # 记录原始视频文件名
    has_violations = db.Column(db.Boolean, default=False, nullable=False) # 记录是否有违规内容 (True/False)
    report_summary = db.Column(db.Text, nullable=True) # 存储风险部分的摘要，方便快速预览
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) # 记录调用的时间

    def to_dict(self):
        """辅助函数：将模型对象转换为字典，方便返回JSON"""
        return {
            "id": self.id,
            "user_ip": self.user_ip,
            "video_filename": self.video_filename,
            "has_violations": self.has_violations,
            "report_summary": self.report_summary,
            "timestamp": self.timestamp.isoformat() + "Z" # 使用ISO格式并加上Z表示UTC时间
        }

@app.route('/download', methods=['POST'])
def download_video_endpoint():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "请求中未包含 URL"}), 400

    video_url = data['url']
    print(f"收到下载请求，URL: {video_url}")

    try:
        # 定义下载选项，文件名将包含视频标题
        output_template = os.path.join(DOWNLOAD_FOLDER, '%(title)s.%(ext)s')
        
        ydl_opts = {
            'outtmpl': output_template,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # 优先下载 mp4
            'merge_output_format': 'mp4',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            # 获取下载后的实际文件名
            downloaded_file_path = ydl.prepare_filename(info_dict)

        print(f"视频下载成功: {downloaded_file_path}")
        
        return jsonify({
            "message": "下载成功", 
            "filename": os.path.basename(downloaded_file_path),
            "original_title": info_dict.get('title', 'N/A')
        })

    except Exception as e:
        print(f"yt-dlp 下载失败: {e}")
        return jsonify({"error": f"视频下载失败: {str(e)}"}), 500

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_video_endpoint():
    # ---------------------------------------------
    # 第一部分：处理 GET 请求
    # ---------------------------------------------
    if request.method == 'GET':
        # 如果是 GET 请求，就返回一个提示页面，然后函数结束。
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>视频分析接口</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding: 40px; }
                h1 { color: #333; }
                p { color: #666; }
                code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>这是一个视频分析的后端接口</h1>
            <p>您不能通过浏览器直接访问这个地址来上传视频。</p>
            <p>请通过我们的前端页面来使用文件上传功能。</p>
            <p>这个地址 (<code>/analyze</code>) 只接受 <code>POST</code> 方法提交的视频文件。</p>
        </body>
        </html>
        """

    # ---------------------------------------------
    # 第二部分：处理 POST 请求
    # ---------------------------------------------
    # 如果代码能执行到这里，说明 request.method 一定是 'POST'
     # --- 【核心修改】处理 POST 请求的逻辑 ---
    user_ip = request.remote_addr
    is_temp_file = False # 标记是否是临时文件，分析完要删除

    # 判断是文件上传，还是基于已下载的文件名进行分析
    if 'video' in request.files and request.files['video'].filename != '':
        # 1. 处理文件上传 (原始逻辑)
        video_file = request.files['video']
        video_filename = video_file.filename
        print(f"收到来自 IP [{user_ip}] 的文件上传请求，处理文件: {video_filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            video_path = tmp.name
        is_temp_file = True # 标记为临时文件

    elif 'filename' in request.form:
        # 2. 处理基于已下载文件的分析请求 (新逻辑)
        video_filename = request.form.get('filename')
        print(f"收到来自 IP [{user_ip}] 的分析请求，处理已下载文件: {video_filename}")
        
        video_path = os.path.join(DOWNLOAD_FOLDER, video_filename)
        if not os.path.exists(video_path):
            return jsonify({"error": "服务器上找不到指定的已下载文件"}), 404
        
        # downloaded files are not temporary
        is_temp_file = False 

    else:
        return jsonify({"error": "请求无效，既没有上传文件也没有提供文件名"}), 400
    
    try:
        params = {
            'useSceneDetection': request.form.get('useSceneDetection', 'true').lower() == 'true',
            'sceneThreshold': float(request.form.get('sceneThreshold', 27.0)),
            'useAdaptive': request.form.get('useAdaptive', 'false').lower() == 'true',
            'chooseFps': float(request.form.get('chooseFps', 1.5)),
            'maxSide': int(request.form.get('maxSide', 512)),
            'timeScale': float(request.form.get('timeScale', 0.1)),
            'promptText': request.form.get('promptText', ''),
            'enableAsr': request.form.get('enableAsr', 'true').lower() == 'true',
            'asrLanguage': request.form.get('asrLanguage', ''),
            'asrBeamSize': int(request.form.get('asrBeamSize', 5)),
            'asrVad': request.form.get('asrVad', 'true').lower() == 'true',
            'asrMaxChars': int(request.form.get('asrMaxChars', 1200)),
            'maxFramesPerChunk': int(request.form.get('maxFramesPerChunk', 30)),
            'llmApiBase': request.form.get('llmApiBase', '').strip(),
            'llmApiKey': request.form.get('llmApiKey', '').strip(),
            'llmModelName': request.form.get('llmModelName', '').strip(),
        }

        results = run_full_analysis(video_path, params)
        
        with app.app_context():
            overall_report = results.get("report", "")
            has_violations, risk_summary = determine_violation_status(overall_report)

            new_log = AnalysisLog(
                user_ip=user_ip,
                video_filename=video_filename,
                has_violations=has_violations,
                report_summary=risk_summary
            )
            
            db.session.add(new_log)
            db.session.commit()
            print(f"新记录已存入数据库: ID={new_log.id}, IP={user_ip}, 文件名='{video_filename}', 违规={has_violations}")
        
        return jsonify(results)

    except Exception as e:
        print(f"处理请求时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/logs', methods=['GET'])
def get_logs():
    """返回数据库中所有的分析日志，按时间倒序排列。"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    with app.app_context():
        # 使用分页查询，提高性能
        pagination = AnalysisLog.query.order_by(AnalysisLog.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
        logs = pagination.items
        return jsonify({
            "total": pagination.total,
            "pages": pagination.pages,
            "current_page": page,
            "logs": [log.to_dict() for log in logs]
        })
