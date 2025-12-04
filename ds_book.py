import os
import json
import time
import requests
import re

# ================= 配置区域 =================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "这里填写你的api-key")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-reasoner"

SOURCE_DIR = "./how-to-optimize-gemm"
OUTPUT_FILENAME = "The_Perfect_GEMM_Book.md"
CACHE_FILE = "ai_explanations_cache_full.json"

# 要排除的文件夹
EXCLUDED_DIRS = ['.git', 'figures', os.path.join('src', 'HowToOptimizeGemm', 'venv')]

# 要排除的文件扩展名
EXCLUDED_EXTENSIONS = ('.m', '.o')

# ================= 1. API 模块 =================
class AIExplainer:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except: pass

    def get_explanation(self, filename, code_content):
        if filename in self.cache:
            print(f"[Cache] {filename}")
            return self.cache[filename]

        print(f"[API] Analyzing {filename}...")

        # 上下文注入
        special_context = ""
        if "MMult_4x4_14.c" in filename:
            special_context = "CRITICAL: Explain that this version unnecessarily packs B every time, causing overhead."
        elif "MMult_4x4_15.c" in filename:
            special_context = "CRITICAL: Explain how 'static packedB' and 'first_time' amortize the packing cost."

        system_prompt = "You are an HPC Professor writing a textbook."
        user_prompt = (
            f"Analyze '{filename}'. {special_context}\nCode Snippet (first 10k chars):\n```\n{code_content[:10000]}\n```\n"
            "Write a Markdown explanation:\n"
            "# Role & Purpose\nOne sentence summary.\n"
            "# Technical Mechanism\nKey logic explanation.\n"
            "# HPC Context\nWhy this matters for performance.\n\n"
            "Rules:\n- Use **bold** for key terms.\n- Use bullet points.\n- No LaTeX math ($), use text (A * B)."
        )

        retries = 3
        for i in range(retries):
            try:
                response = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={"model": MODEL_NAME, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.0},
                    timeout=120
                )
                response.raise_for_status()
                explanation = response.json()['choices'][0]['message']['content']
                self.cache[filename] = explanation
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                time.sleep(1)
                return explanation
            except Exception as e:
                print(f"Retry {i+1} error: {e}")
                time.sleep(2)
        return "Error: Could not retrieve explanation."

# ================= 2. 工具函数 =================
def sanitize_text(text):
    replacements = {'\u2013': '-', '\u2014': '--', '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2022': '*', '\t': '    '}
    for k, v in replacements.items(): text = text.replace(k, v)
    # 移除 LaTeX
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    return text

def is_excluded_path(path, source_dir):
    """检查路径是否在排除列表中"""
    rel_path = os.path.relpath(path, source_dir)
    rel_path_parts = rel_path.split(os.sep)

    # 检查路径是否以任何排除目录开头
    for excluded in EXCLUDED_DIRS:
        excluded_parts = excluded.split(os.sep)
        # 如果路径的前几部分与排除目录匹配
        if len(rel_path_parts) >= len(excluded_parts):
            if all(rel_path_parts[i] == excluded_parts[i] for i in range(len(excluded_parts))):
                return True

    return False

def get_all_files(source_dir):
    """自动扫描所有相关文件并智能分类，排除特定文件夹和文件类型"""
    file_groups = {
        "Overview": [],
        "Infrastructure": [],
        "Baseline": [],
        "Scalar Opt (1x4)": [],
        "Vectorization (4x4 SIMD)": [],
        "Memory Opt (Packing)": []
    }

    # 要包含的文件扩展名
    extensions = ('.c', '.h', '.py', '.md', '.txt', '.sh')

    for root, dirs, files in os.walk(source_dir):
        # 排除特定目录
        dirs[:] = [d for d in dirs if not is_excluded_path(os.path.join(root, d), source_dir)]

        for f in files:
            # 检查文件扩展名是否在排除列表中
            if f.lower().endswith(EXCLUDED_EXTENSIONS):
                continue

            # 检查文件是否在允许的扩展名列表中或是指定的特殊文件
            if f.lower().endswith(extensions) or f.lower() in ['makefile', 'cmakelists.txt']:
                path = os.path.join(root, f)

                # 检查是否在排除路径中
                if is_excluded_path(path, source_dir):
                    continue

                # 跳过隐藏文件
                if f.startswith('.'):
                    continue

                # 智能分类逻辑
                if f.lower() == "readme.md":
                    file_groups["Overview"].append((f, path))
                elif "MMult0" in f or "MMult1" in f or "MMult2" in f:
                    file_groups["Baseline"].append((f, path))
                elif "MMult_1x4" in f:
                    file_groups["Scalar Opt (1x4)"].append((f, path))
                elif "MMult_4x4" in f:
                    # 区分向量化和内存优化
                    if any(x in f for x in ['11','12','13','14','15']):
                        file_groups["Memory Opt (Packing)"].append((f, path))
                    else:
                        file_groups["Vectorization (4x4 SIMD)"].append((f, path))
                else:
                    # 其他工具文件
                    file_groups["Infrastructure"].append((f, path))

    # 排序
    for k in file_groups:
        file_groups[k].sort(key=lambda x: x[0])

    return file_groups

# ================= 3. Markdown 生成类 =================
class MarkdownBook:
    def __init__(self):
        self.content = ""
        
    def add_header(self, text, level=1):
        """添加标题"""
        self.content += f"{'#' * level} {text}\n\n"
    
    def add_text(self, text):
        """添加普通文本"""
        self.content += f"{text}\n\n"
    
    def add_code_block(self, code, language="c"):
        """添加代码块"""
        self.content += f"```{language}\n{code}\n```\n\n"
    
    def add_horizontal_line(self):
        """添加水平线"""
        self.content += "---\n\n"
    
    def add_file_section(self, filename, explanation, code_content):
        """添加文件章节"""
        self.add_header(f"File: {filename}", level=2)
        self.add_text(explanation)
        self.add_header("Source Code Implementation", level=3)
        self.add_code_block(code_content)
        self.add_horizontal_line()
    
    def save(self, filename):
        """保存Markdown文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.content)
        print(f"[Success] Markdown file saved: {filename}")

# ================= 4. 主程序 =================
def generate_book():
    ai = AIExplainer(CACHE_FILE)
    md = MarkdownBook()
    
    # 封面/标题
    md.add_header("How To Optimize GEMM")
    md.add_text("The Ultimate Edition")
    md.add_text("Including analysis of ALL project files")
    md.add_horizontal_line()
    
    # 目录
    md.add_header("Table of Contents", level=2)
    md.add_text("""
1. [Overview](#overview)
2. [Infrastructure](#infrastructure)
3. [Baseline](#baseline)
4. [Scalar Opt (1x4)](#scalar-opt-1x4)
5. [Vectorization (4x4 SIMD)](#vectorization-4x4-simd)
6. [Memory Opt (Packing)](#memory-opt-packing)
    """)
    md.add_horizontal_line()
    
    # 获取所有文件并分类
    file_groups = get_all_files(SOURCE_DIR)
    
    # 打印统计信息
    total_files = sum(len(files) for files in file_groups.values())
    print(f"\n[Info] Total files found: {total_files}")
    for category, files in file_groups.items():
        if files:
            print(f"  - {category}: {len(files)} files")
    
    # 定义章节顺序
    chapter_order = [
        ("Overview", "overview"),
        ("Infrastructure", "infrastructure"),
        ("Baseline", "baseline"),
        ("Scalar Opt (1x4)", "scalar-opt-1x4"),
        ("Vectorization (4x4 SIMD)", "vectorization-4x4-simd"),
        ("Memory Opt (Packing)", "memory-opt-packing")
    ]
    
    for idx, (chapter_name, chapter_id) in enumerate(chapter_order):
        files = file_groups.get(chapter_name, [])
        if not files:
            print(f"\n[Info] No files found for chapter: {chapter_name}")
            continue
        
        print(f"\n[Info] Processing chapter {idx+1}: {chapter_name} ({len(files)} files)")
        md.add_header(f"Chapter {idx+1}: {chapter_name}", level=1)
        md.add_header(f"<a id='{chapter_id}'></a>", level=2)
        
        for fname, fpath in files:
            try:
                print(f"  - Processing: {fname}")
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                # AI 解释
                expl = ai.get_explanation(fname, code)
                # 添加到 Markdown
                md.add_file_section(fname, expl, code)
                
            except Exception as e:
                print(f"  [Error] Skipping {fname}: {e}")
                md.add_header(f"File: {fname} (Error)", level=2)
                md.add_text(f"Error reading file: {e}")
                md.add_horizontal_line()
    
    # 保存Markdown文件
    md.save(OUTPUT_FILENAME)
    print(f"\n[Success] Book generated: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate_book()
