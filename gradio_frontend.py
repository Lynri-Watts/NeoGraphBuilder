import gradio as gr
import requests
import time
import json
import logging
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API基础URL
API_BASE_URL = "http://localhost:8000"

# 任务状态映射
STATUS_MAP = {
    "pending": "等待中",
    "processing": "处理中",
    "completed": "已完成",
    "failed": "失败"
}

# 提交任务函数
def submit_task(topic, use_latex, prompt_supplement, max_documents=10):
    """
    提交文献综述生成任务到后端API
    """
    # 表单验证
    if not topic or not topic.strip():
        return None, "请输入有效的研究主题", None
    
    try:
        # 构建请求数据
        data = {
            "topic": topic.strip(),
            "use_latex": use_latex,
            "prompt_supplement": prompt_supplement.strip() if prompt_supplement else "",
            "max_documents": max_documents
        }
        
        # 发送POST请求到API
        response = requests.post(
            f"{API_BASE_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=10  # 添加超时设置
        )
        
        response.raise_for_status()  # 检查HTTP错误
        result = response.json()
        
        # 返回任务ID、初始状态和LaTeX标志
        return result["task_id"], "任务已提交，正在处理中...", use_latex
    except requests.exceptions.ConnectionError:
        error_message = "连接错误：无法连接到后端服务。请确认API服务正在运行。"
        logger.error(error_message)
        return None, error_message, None
    except requests.exceptions.Timeout:
        error_message = "请求超时：连接后端服务超时。"
        logger.error(error_message)
        return None, error_message, None
    except Exception as e:
        error_message = f"提交任务失败: {str(e)}"
        logger.error(error_message)
        return None, error_message, None

# 查询任务状态函数
def check_task_status(task_id):
    """
    查询任务的处理状态
    """
    if not task_id:
        return False, "请先提交任务"
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/status/{task_id}",
            timeout=5  # 添加超时设置
        )
        response.raise_for_status()
        status = response.json()
        
        # 获取中文状态描述
        status_text = STATUS_MAP.get(status["status"], status["status"])
        return status["completed"], status_text
    except requests.exceptions.ConnectionError:
        error_message = "连接错误：无法连接到后端服务。"
        logger.error(error_message)
        return False, error_message
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False, "任务不存在或已过期"
        return False, f"查询状态失败: {str(e)}"
    except Exception as e:
        error_message = f"查询状态失败: {str(e)}"
        logger.error(error_message)
        return False, error_message

# 获取任务结果函数
def get_task_result(task_id, is_latex=False):
    """
    获取任务的生成结果
    """
    if not task_id:
        return "请先提交任务"
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/result/{task_id}",
            timeout=10  # 添加超时设置
        )
        result = response.json()
        
        if result.get("error"):
            return f"生成失败: {result['error']}"
        else:
            content = result.get("result", "无结果返回")
            # 如果是LaTeX格式，可以添加适当的提示
            if is_latex and content and "无结果" not in content:
                content = "[LaTeX格式输出]\n" + content
            return content
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return "任务不存在或已过期"
        elif e.response.status_code == 409:
            return "任务尚未完成，请继续等待"
        else:
            error_message = f"获取结果失败: {str(e)}"
            logger.error(error_message)
            return error_message
    except Exception as e:
        error_message = f"获取结果失败: {str(e)}"
        logger.error(error_message)
        return error_message

# 轮询任务状态并获取结果的函数（适合后台线程）
def wait_for_result_background(task_id, status_update_fn, result_update_fn, max_wait_time=300):
    """
    在后台线程中轮询任务状态并更新UI
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        completed, status = check_task_status(task_id)
        
        # 更新状态
        elapsed_time = int(time.time() - start_time)
        status_text = f"{status} (已等待{elapsed_time}秒)"
        status_update_fn(status_text)
        
        if completed or status == "失败":
            # 任务完成或失败，获取结果
            result = get_task_result(task_id)
            result_update_fn(result)
            break
        
        # 等待2秒后再次查询
        time.sleep(2)
    
    if time.time() - start_time >= max_wait_time:
        status_update_fn("查询超时，请稍后手动查询结果")

# 简化的自动等待结果函数
def auto_wait_for_result(task_id, is_latex, max_documents=10):
    """
    自动等待结果的简化实现，支持最大文献数参数
    """
    if not task_id:
        gr.Warning("请先提交任务")
        return "请先提交任务", ""
    
    # 检查后端服务是否可访问
    try:
        requests.get(f"{API_BASE_URL}/", timeout=2)
    except:
        gr.Error("无法连接到后端服务，请确认API服务正在运行")
        return "无法连接到后端服务", ""
    
    gr.Info(f"任务 {task_id} 正在处理中（最大文献数：{max_documents}），请定期点击'检查状态'查看进度")
    return f"任务正在处理中（最大文献数：{max_documents}），请定期点击'检查状态'查看进度", ""

# 后台线程轮询函数
def wait_for_result_background(task_id, status_update_fn, result_update_fn):
    """
    在后台线程中轮询任务状态，更新UI
    """
    try:
        # 最多轮询30次，每次间隔3秒
        for i in range(30):
            time.sleep(3)
            
            try:
                completed, status = check_task_status(task_id)
                status_update_fn(f"{status} (已等待{(i+1)*3}秒)")
                
                if completed or status == "失败":
                    result = get_task_result(task_id, True)
                    result_update_fn(result)
                    return
            except Exception as e:
                status_update_fn(f"查询出错: {str(e)}")
                logger.error(f"后台轮询出错: {str(e)}")
        
        # 超时
        status_update_fn("轮询超时，请手动查询结果")
    except Exception as e:
        logger.error(f"后台轮询线程错误: {str(e)}")

# 主函数：设置Gradio界面
def create_gradio_interface():
    with gr.Blocks(title="文献综述生成器") as interface:
        gr.Markdown("# 文献综述生成器")
        gr.Markdown("输入研究主题，选择输出格式，提交后等待生成结果")
        
        # 存储当前任务的LaTeX设置
        current_use_latex = gr.State(False)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                topic = gr.Textbox(
                    label="研究主题",
                    placeholder="请输入研究主题...",
                    lines=2,
                    info="请提供清晰、具体的研究主题"
                )
                
                use_latex = gr.Checkbox(
                    label="使用LaTeX格式输出",
                    value=False,
                    info="选中后将生成LaTeX格式的文献综述"
                )
                
                # 最大文献数输入
                max_documents_input = gr.Slider(
                    label="最大文献数",
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    interactive=True,
                    info="设置生成时参考的最大文献数量"
                )
                
                prompt_supplement = gr.Textbox(
                    label="额外要求（可选）",
                    placeholder="请输入额外的生成要求...",
                    lines=3,
                    info="您可以指定特定的格式、内容侧重点等要求"
                )
                
                # 按钮区域
                with gr.Row():
                    submit_btn = gr.Button("提交任务", variant="primary", size="sm")
                
                # 状态显示区域
                task_id_output = gr.Textbox(label="任务ID", interactive=False)
                status_output = gr.Textbox(label="处理状态", interactive=False)
                
                # 操作按钮区域
                with gr.Row():
                    check_status_btn = gr.Button("检查状态", size="sm")
                    auto_wait_btn = gr.Button("自动等待结果", size="sm")
                    get_result_btn = gr.Button("获取结果", size="sm")
                
                # 使用说明
                gr.Markdown("""
                ### 使用说明
                1. 输入研究主题和可选的额外要求
                2. 选择是否需要LaTeX格式输出
                3. 点击"提交任务"按钮
                4. 可以选择手动检查状态和获取结果，或点击"自动等待结果"让系统自动跟踪
                """)
            
            with gr.Column(scale=2):
                # 结果显示区域
                result_output = gr.Textbox(
                    label="生成结果",
                    interactive=False,
                    lines=20
                )
                
                # 清空按钮
                clear_btn = gr.Button("清空所有", size="sm")
        
        # 设置按钮点击事件
        submit_btn.click(
            fn=submit_task,
            inputs=[topic, use_latex, prompt_supplement, max_documents_input],
            outputs=[task_id_output, status_output, current_use_latex]
        )
        
        # 添加状态显示更新的函数，用于更好的用户反馈
        def update_status_with_feedback(task_id):
            if not task_id:
                gr.Warning("请先提交任务")
                return "请先提交任务"
            
            try:
                completed, status = check_task_status(task_id)
                if completed:
                    gr.Info(f"任务已完成！状态: {status}")
                elif status == "失败":
                    gr.Error(f"任务失败！状态: {status}")
                return status
            except Exception as e:
                error_msg = f"查询状态时出错: {str(e)}"
                gr.Error(error_msg)
                logger.error(error_msg)
                return error_msg
        
        # 定义获取结果的反馈函数
        def get_result_with_feedback(task_id, is_latex):
            if not task_id:
                gr.Warning("请先提交任务")
                return "请先提交任务"
            
            try:
                # 先检查任务状态
                completed, status = check_task_status(task_id)
                if not completed and status != "失败":
                    gr.Warning("任务尚未完成，请继续等待或使用'自动等待结果'功能")
                    return "任务尚未完成"
                
                # 获取结果
                result = get_task_result(task_id, is_latex)
                if "失败" in result:
                    gr.Error(result)
                else:
                    gr.Info("成功获取结果！")
                return result
            except Exception as e:
                error_msg = f"获取结果时出错: {str(e)}"
                gr.Error(error_msg)
                logger.error(error_msg)
                return error_msg
        
        # 检查状态按钮
        check_status_btn.click(
            fn=update_status_with_feedback,
            inputs=[task_id_output],
            outputs=[status_output]
        )
        
        # 获取结果按钮
        get_result_btn.click(
            fn=get_result_with_feedback,
            inputs=[task_id_output, current_use_latex],
            outputs=[result_output]
        )
        
        # 自动等待结果按钮
        auto_wait_btn.click(
            fn=auto_wait_for_result,
            inputs=[task_id_output, current_use_latex, max_documents_input],
            outputs=[status_output, result_output]
        )
        
        # 清空按钮
        clear_btn.click(
            fn=lambda: ("", "", "", False),
            outputs=[topic, prompt_supplement, task_id_output, current_use_latex]
        )
        
        # 界面加载时的提示
        gr.Markdown("""
        > **注意：** 生成文献综述可能需要较长时间，请耐心等待。任务提交后可以关闭页面，稍后通过任务ID查询结果。
        """)
    
    return interface

# 启动界面
if __name__ == "__main__":
    # 检查API服务是否可访问
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        logger.info("成功连接到后端API服务")
    except Exception as e:
        logger.warning(f"启动时无法连接到后端服务: {str(e)}")
        logger.warning("请确保后端API服务正在运行 (http://localhost:8000)")
    
    # 创建并启动Gradio界面
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # 在生产环境中设置为False
    )