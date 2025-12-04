#!/bin/bash

# ============================================
# GEMM 书籍自动构建脚本
# 功能：检查环境依赖 -> 克隆仓库 -> 生成PDF
# ============================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 重置颜色

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ================= 1. 依赖检查函数 =================
check_dependencies() {
    log_info "开始检查系统依赖..."
    
    local missing_deps=()
    local python_packages=("os" "json" "time" "requests" "re")
    local python_missing_pkgs=()

    # 检查命令行工具
    for cmd in git python3 pandoc; do
        if command -v $cmd &> /dev/null; then
            log_success "$cmd 已安装 ($($cmd --version 2>/dev/null | head -n1))"
        else
            log_error "$cmd 未安装"
            missing_deps+=("$cmd")
        fi
    done

    # 检查 texlive-xetex
    if dpkg -l | grep -q texlive-xetex 2>/dev/null || \
       pacman -Q texlive-bin 2>/dev/null || \
       rpm -qa | grep -q texlive-xetex 2>/dev/null; then
        log_success "texlive-xetex 已安装"
    else
        log_warning "texlive-xetex 未安装 (PDF生成需要)"
        missing_deps+=("texlive-xetex")
    fi

    # 检查Python包
    log_info "检查Python包..."
    python3 -c "
import sys
import importlib
packages = ['os', 'json', 'time', 'requests', 're']
missing = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f'[SUCCESS] Python包 {pkg} 可用')
    except ImportError as e:
        if pkg == 're':
            print(f'[WARNING] Python包 {pkg} 是标准库，但导入失败: {e}')
        else:
            print(f'[ERROR] Python包 {pkg} 未安装')
            missing.append(pkg)
if missing:
    sys.exit(1)
" 2>&1 | while read line; do
        if [[ $line == *"[ERROR]"* ]]; then
            echo -e "${RED}$line${NC}"
            python_missing_pkgs+=("$(echo $line | awk '{print $4}')")
        elif [[ $line == *"[WARNING]"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        else
            echo -e "${GREEN}$line${NC}"
        fi
    done

    # 汇总检查结果
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_warning "缺失系统包: ${missing_deps[*]}"
        return 1
    fi
    
    if [ ${#python_missing_pkgs[@]} -gt 0 ]; then
        log_warning "缺失Python包: ${python_missing_pkgs[*]}"
        return 2
    fi
    
    log_success "所有依赖检查通过！"
    return 0
}

# ================= 2. 依赖安装函数 =================
install_dependencies() {
    log_info "尝试安装缺失依赖..."
    
    # 检测包管理器
    if command -v apt &> /dev/null; then
        PKG_MANAGER="apt"
        INSTALL_CMD="sudo apt install -y"
    elif command -v pacman &> /dev/null; then
        PKG_MANAGER="pacman"
        INSTALL_CMD="sudo pacman -S --noconfirm"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        INSTALL_CMD="sudo yum install -y"
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
        INSTALL_CMD="sudo dnf install -y"
    else
        log_error "无法识别包管理器"
        return 1
    fi
    
    log_info "使用包管理器: $PKG_MANAGER"
    
    # 安装缺失的系统包
    for pkg in git python3 pandoc texlive-xetex; do
        if ! command -v $pkg &> /dev/null && [ "$pkg" != "texlive-xetex" ]; then
            log_info "安装 $pkg..."
            $INSTALL_CMD $pkg
        fi
    done
    
    # 特殊处理 texlive-xetex
    if ! (dpkg -l | grep -q texlive-xetex 2>/dev/null) && [ "$PKG_MANAGER" = "apt" ]; then
        log_info "安装 texlive-xetex..."
        $INSTALL_CMD texlive-xetex
    fi
    
    # 安装Python包
    for pkg in requests; do
        python3 -c "import $pkg" 2>/dev/null || {
            log_info "安装Python包: $pkg"
            pip3 install $pkg
        }
    done
    
    log_success "依赖安装完成"
    return 0
}

# ================= 3. 构建流程函数 =================
build_process() {
    local repo_dir="how-to-optimize-gemm"
    
    # 步骤1: 克隆仓库
    log_info "步骤1: 克隆 GitHub 仓库..."
    if [ -d "$repo_dir" ]; then
        log_warning "目录 '$repo_dir' 已存在，尝试更新..."
        cd "$repo_dir" && git pull && cd - || {
            log_error "无法更新仓库"
            return 1
        }
    else
        git clone https://github.com/flame/how-to-optimize-gemm.git || {
            log_error "克隆仓库失败"
            return 1
        }
    fi
    log_success "仓库准备完成"

    # 步骤2: 运行Python脚本
    log_info "步骤2: 运行 ds_book.py..."
    if [ -f "ds_book.py" ]; then
        python3 ds_book.py || {
            log_error "运行 ds_book.py 失败"
            return 1
        }
    elif [ -f "$repo_dir/ds_book.py" ]; then
        cd "$repo_dir"
        python3 ds_book.py || {
            log_error "运行 ds_book.py 失败"
            cd -
            return 1
        }
        cd -
    else
        log_error "找不到 ds_book.py"
        log_info "在以下位置查找:"
        find . -name "ds_book.py" -type f 2>/dev/null || echo "未找到"
        return 1
    fi
    log_success "Python脚本执行完成"

    # 步骤3: 生成PDF
    log_info "步骤3: 生成PDF文档..."
    if [ -f "The_Perfect_GEMM_Book.md" ]; then
        pandoc The_Perfect_GEMM_Book.md --pdf-engine=xelatex -o gemm.pdf || {
            log_warning "标准PDF生成失败，尝试使用数学字体..."
            pandoc The_Perfect_GEMM_Book.md --pdf-engine=xelatex \
                -V mainfont="Libertinus Serif" \
                -V mathfont="Libertinus Math" \
                -o gemm.pdf || {
                log_error "PDF生成失败"
                return 1
            }
        }
    else
        log_error "找不到 The_Perfect_GEMM_Book.md"
        log_info "在以下位置查找Markdown文件:"
        find . -name "*.md" -type f 2>/dev/null | head -5
        return 1
    fi
    
    # 验证PDF生成
    if [ -f "gemm.pdf" ]; then
        file_size=$(du -h gemm.pdf | cut -f1)
        log_success "PDF生成成功！文件: gemm.pdf (大小: $file_size)"
        
        # 尝试打开PDF（如果支持）
        if command -v xdg-open &> /dev/null; then
            read -p "是否要打开生成的PDF？(y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                xdg-open gemm.pdf &
            fi
        fi
    else
        log_error "PDF文件未生成"
        return 1
    fi
    
    return 0
}

# ================= 4. 主函数 =================
main() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}    GEMM 书籍自动构建脚本    ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    
    # 检查依赖
    check_dependencies
    local dep_status=$?
    
    if [ $dep_status -ne 0 ]; then
        log_warning "依赖不完整"
        read -p "是否尝试自动安装缺失依赖？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_dependencies
            # 重新检查
            check_dependencies
            if [ $? -ne 0 ]; then
                log_error "依赖安装后仍不满足，请手动安装"
                exit 1
            fi
        else
            log_error "请手动安装缺失依赖后重新运行脚本"
            exit 1
        fi
    fi
    
    # 执行构建流程
    log_info "开始构建流程..."
    if build_process; then
        echo -e "${GREEN}=========================================${NC}"
        echo -e "${GREEN}     构建成功完成！🎉     ${NC}"
        echo -e "${GREEN}=========================================${NC}"
        echo -e "生成的PDF: ${YELLOW}$(pwd)/gemm.pdf${NC}"
    else
        echo -e "${RED}=========================================${NC}"
        echo -e "${RED}     构建失败！😞     ${NC}"
        echo -e "${RED}=========================================${NC}"
        exit 1
    fi
}

# ================= 5. 脚本执行 =================
# 设置错误时退出
set -e

# 捕获中断信号
trap 'log_error "脚本被用户中断"; exit 1' INT TERM

# 运行主函数
main "$@"
