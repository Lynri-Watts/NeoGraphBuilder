import asyncio
import uuid
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入文献综述生成器
from literature_review_generator import LiteratureReviewGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="文献综述生成API",
    description="提供异步文献综述生成服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务状态枚举
class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 任务数据模型
class Task:
    def __init__(self, task_id: str, topic: str, use_latex: bool, prompt_supplement: str, max_documents: int = 10, max_workers: int = 4):
        self.task_id = task_id
        self.topic = topic
        self.use_latex = use_latex
        self.prompt_supplement = prompt_supplement
        self.max_documents = max_documents
        self.max_workers = max_workers
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None

# 存储所有任务的字典
tasks: Dict[str, Task] = {}

# 请求模型
class GenerateRequest(BaseModel):
    topic: str
    use_latex: bool = False
    prompt_supplement: Optional[str] = ""
    max_documents: Optional[int] = 10
    max_workers: int = 4  # 并行处理文档的最大工作线程数

# 响应模型
class TaskResponse(BaseModel):
    task_id: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    completed: bool

class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "文献综述生成API服务运行中",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "status": "/api/status/{task_id}",
            "result": "/api/result/{task_id}"
        }
    }

@app.post("/api/generate", response_model=TaskResponse)
async def generate_review(request: GenerateRequest, background_tasks: BackgroundTasks):
    """提交文献综述生成任务"""
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务
    task = Task(
        task_id=task_id,
        topic=request.topic,
        use_latex=request.use_latex,
        prompt_supplement=request.prompt_supplement,
        max_documents=request.max_documents,
        max_workers=request.max_workers
    )
    
    # 存储任务
    tasks[task_id] = task
    
    # 异步执行任务
    background_tasks.add_task(process_review_task, task_id)
    
    logger.info(f"创建新任务: {task_id}, 主题: {request.topic}, LaTeX: {request.use_latex}, 最大文献数: {request.max_documents}, 并行工作线程数: {request.max_workers}")
    
    return TaskResponse(task_id=task_id, status=TaskStatus.PENDING)

@app.get("/api/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task.status,
        completed=task.status == TaskStatus.COMPLETED
    )

@app.get("/api/result/{task_id}", response_model=TaskResultResponse)
async def get_task_result(task_id: str):
    """获取任务结果"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status == TaskStatus.PENDING or task.status == TaskStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="任务尚未完成")
    
    return TaskResultResponse(
        task_id=task_id,
        status=task.status,
        result=task.result,
        error=task.error
    )

async def process_review_task(task_id: str):
    """异步处理文献综述生成任务"""
    task = tasks.get(task_id)
    if not task:
        return
    
    task.status = TaskStatus.PROCESSING
    
    try:
        logger.info(f"开始处理任务: {task_id}")
        
        # 创建文献综述生成器实例（不生成报告文件）
        generator = LiteratureReviewGenerator(generate_reports=False)
        
        # 生成文献综述
        result = await asyncio.to_thread(
            generator.generate_review,
            task.topic,
            task.use_latex,
            prompt_supplement=task.prompt_supplement,
            max_documents=task.max_documents,
            max_workers=task.max_workers
        )
        
        # 保存结果
        task.result = result
        task.status = TaskStatus.COMPLETED
        logger.info(f"任务完成: {task_id}")
        
    except Exception as e:
        error_message = str(e)
        task.error = error_message
        task.status = TaskStatus.FAILED
        logger.error(f"任务失败: {task_id}, 错误: {error_message}")
    finally:
        # 清理资源
        try:
            generator.close()
        except:
            pass

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )